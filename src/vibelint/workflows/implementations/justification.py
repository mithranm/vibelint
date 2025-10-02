"""Clean, focused justification engine.

Core workflow:
1. Discover files and build filesystem tree
2. Summarize each file's purpose with fast LLM (cached by hash)
3. Generate XML context with structure + summaries
4. Orchestrator LLM analyzes for misplaced/useless/redundant files
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from vibelint.workflows.core.base import WorkflowResult

from vibelint.workflows.core.base import BaseWorkflow, WorkflowConfig

logger = logging.getLogger(__name__)


class JustificationEngine(BaseWorkflow):
    """Clean justification engine focused on the essential workflow."""

    # Workflow metadata for registry
    workflow_id: str = "justification"
    name: str = "Code Justification Analysis"
    description: str = "Analyze code architecture for misplaced/useless/redundant files using LLM"
    version: str = "3.0"
    category: str = "analysis"
    tags: set = {"code-quality", "llm-analysis", "architecture"}

    def __init__(self, config: Optional[WorkflowConfig] = None):
        super().__init__(config)
        self.config = config or WorkflowConfig()
        self.llm_client = None
        self.cache_file = Path(".vibes/cache/file_summaries.json")
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.xml_output = Path(f".vibes/reports/project_analysis_{self.timestamp}.xml")
        self.jsonl_log_file = None
        self.master_log_file = None
        self.xml_file = None

        # Initialize LLM manager
        self._init_llm()

        # Set up logging (both JSONL for LLM calls and master log for process)
        self._setup_logging()

        # Load summary cache
        self.summary_cache = self._load_cache()

    def _init_llm(self):
        """Initialize LLM client and get token limits from config."""
        try:
            from vibelint.llm_client import LLMClient, LLMRequest

            self.llm_client = LLMClient()
            self.LLMRequest = LLMRequest

            # Get token limits from LLM config for proper chunking
            self.fast_max_tokens = self.llm_client.llm_config.fast_max_tokens
            self.fast_max_context_tokens = self.llm_client.llm_config.fast_max_context_tokens or (
                self.fast_max_tokens * 4
            )
            self.orchestrator_max_tokens = self.llm_client.llm_config.orchestrator_max_tokens
            self.orchestrator_max_context_tokens = (
                self.llm_client.llm_config.orchestrator_max_context_tokens or 131072
            )

            logger.info(
                f"LLM manager initialized (fast: {self.fast_max_tokens} output tokens, {self.fast_max_context_tokens} context tokens, orchestrator: {self.orchestrator_max_tokens} tokens)"
            )
        except Exception as e:
            logger.warning(f"LLM manager not available: {e}")
            # Fallback defaults if LLM not available
            self.fast_max_tokens = 2048
            self.fast_max_context_tokens = 8192
            self.orchestrator_max_tokens = 8192
            self.orchestrator_max_context_tokens = 131072

    def _setup_logging(self):
        """Set up both JSONL (LLM calls) and master log (process) files."""
        logs_dir = Path(".vibes/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Auto-create .gitignore in .vibes/ so outputs don't get tracked
        vibes_gitignore = Path(".vibes/.gitignore")
        if not vibes_gitignore.exists():
            vibes_gitignore.write_text("*\n")

        # Use the same timestamp as the report files for consistency
        # Master log file for human-readable process logging
        self.master_log_file = logs_dir / f"justification_{self.timestamp}.log"

        # JSONL log file for LLM prompt-response pairs
        if self.llm_client:
            self.jsonl_log_file = logs_dir / f"justification_{self.timestamp}.jsonl"
            # Create the file immediately so it always exists
            self.jsonl_log_file.touch()

            # Register callback with LLM manager to log all requests/responses
            def log_callback(log_entry):
                """Write log entry to JSONL file."""
                if self.jsonl_log_file is None:
                    return
                try:
                    # Convert LogEntry dataclass to dict for JSON serialization
                    from dataclasses import asdict

                    log_dict = (
                        asdict(log_entry)
                        if hasattr(log_entry, "__dataclass_fields__")
                        else log_entry
                    )
                    with open(self.jsonl_log_file, "a") as f:
                        f.write(json.dumps(log_dict) + "\n")
                except Exception as e:
                    logger.debug(f"Failed to write JSONL log: {e}")

            self.llm_client.set_log_callback(log_callback)
            self._log(f"JSONL logging enabled: {self.jsonl_log_file}")

        self._log(f"Master log file: {self.master_log_file}")

    def _log(self, message: str):
        """Write to master log file with timestamp."""
        if self.master_log_file:
            try:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(self.master_log_file, "a") as f:
                    f.write(f"[{timestamp}] {message}\n")
            except Exception as e:
                logger.debug(f"Failed to write master log: {e}")
        # Also log to Python logger
        logger.info(message)

    def _load_cache(self) -> Dict[str, str]:
        """Load file summary cache."""
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text())
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}

    def _save_cache(self):
        """Save file summary cache."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(self.summary_cache, indent=2))

    def _get_file_hash(self, file_path: Path) -> str:
        """Get file content hash for caching."""
        try:
            content = file_path.read_text(encoding="utf-8")
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        except Exception:
            return hashlib.sha256(str(file_path).encode()).hexdigest()[:16]

    def _discover_files(self, root: Path) -> List[Path]:
        """Discover all relevant files using vibelint's discovery system."""
        from vibelint.config import load_config
        from vibelint.discovery import discover_files

        # Load config for the project
        config = load_config(root)

        # Use vibelint's discover_files with the project root
        files = discover_files(paths=[root], config=config, explicit_exclude_paths=set())

        # Filter out very large files
        return sorted([f for f in files if f.stat().st_size < 2 * 1024 * 1024])

    def _build_tree_xml(self, root: Path, files: List[Path]) -> ET.Element:
        """Build XML tree structure."""
        project_elem = ET.Element("project", name=root.name, path=str(root))

        # Group files by directory
        dir_structure = {}
        for file_path in files:
            relative = file_path.relative_to(root)
            parts = relative.parts

            current = dir_structure
            for part in parts[:-1]:  # All but the filename
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Add the file
            filename = parts[-1]
            current[filename] = file_path

        def add_to_xml(parent_elem, structure, base_path=""):
            for name, content in sorted(structure.items()):
                if isinstance(content, dict):
                    # It's a directory
                    dir_elem = ET.SubElement(
                        parent_elem, "directory", name=name, path=f"{base_path}/{name}".strip("/")
                    )
                    add_to_xml(dir_elem, content, f"{base_path}/{name}".strip("/"))
                else:
                    # It's a file
                    file_path = content
                    file_elem = ET.SubElement(
                        parent_elem,
                        "file",
                        name=name,
                        path=str(file_path.relative_to(root)),
                        size=str(file_path.stat().st_size),
                    )

                    # Add file summary if available
                    file_hash = self._get_file_hash(file_path)
                    summary = self.summary_cache.get(file_hash, "")
                    if summary:
                        summary_elem = ET.SubElement(file_elem, "summary")
                        summary_elem.text = summary

        add_to_xml(project_elem, dir_structure)
        return project_elem

    def _chunk_content(self, content: str, max_chars: int) -> List[str]:
        """Chunk content to fit in LLM context window."""
        if len(content) <= max_chars:
            return [content]

        chunks = []
        lines = content.split("\n")
        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            if current_size + line_size > max_chars and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def _summarize_file(self, file_path: Path) -> str:
        """Summarize file purpose using fast LLM with proper chunking."""
        from vibelint.filesystem import is_binary

        if not self.llm_client:
            return "[LLM not available]"

        # Check if file is binary
        if is_binary(file_path):
            return "[Binary file - cannot summarize content]"

        try:
            content = file_path.read_text(encoding="utf-8")
            if not content.strip():
                return "[Empty file]"

            # Calculate max content size for fast LLM context window
            # Use 3 chars/token estimation, reserve space for prompt overhead (~100 tokens)
            prompt_overhead_tokens = 100
            max_content_tokens = self.fast_max_context_tokens - prompt_overhead_tokens
            max_content_chars = max_content_tokens * 3

            chunks = self._chunk_content(content, max_content_chars)

            # Analyze each chunk and concatenate summaries (no synthesis step for speed)
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                if len(chunks) == 1:
                    chunk_prompt = f"""File: {file_path.name}

```
{chunk}
```

Output a 1-2 sentence summary of what this file does. No preamble, just the summary:"""
                else:
                    chunk_prompt = f"""File: {file_path.name} (part {i+1}/{len(chunks)})

```
{chunk}
```

One sentence describing what this code does:"""

                request = self.LLMRequest(
                    content=chunk_prompt, max_tokens=self.fast_max_tokens, temperature=0.1
                )

                # LLM manager automatically cascades to orchestrator if fast fails
                response = self.llm_client.process_request_sync(request)

                if response and response.success and response.content:
                    chunk_summaries.append(response.content.strip())
                else:
                    chunk_summaries.append(f"[Chunk {i+1} analysis failed]")

            # Concatenate all chunk summaries
            return " ".join(chunk_summaries)

        except Exception as e:
            logger.warning(f"Failed to summarize {file_path}: {e}")
            return f"[Error: {e}]"

    def _chunk_xml_semantically(self, xml_content: str) -> List[str]:
        """Chunk XML content by directory structure to fit in orchestrator context."""
        # Parse XML to extract file entries
        root = ET.fromstring(xml_content)

        # Group files by their top-level directory
        dir_groups = {}
        for file_elem in root.findall(".//file"):
            file_path = file_elem.get("path", "")
            parts = file_path.split("/")
            top_dir = parts[0] if len(parts) > 1 else "root"

            if top_dir not in dir_groups:
                dir_groups[top_dir] = []

            # Serialize this file element
            file_xml = ET.tostring(file_elem, encoding="unicode")
            dir_groups[top_dir].append(file_xml)

        # Calculate max chunk size (90% of orchestrator context for prompt overhead)
        max_chunk_tokens = int(self.orchestrator_max_context_tokens * 0.9)
        max_chunk_chars = max_chunk_tokens * 3

        # Build chunks from directory groups
        chunks = []
        current_chunk = []
        current_size = 0

        project_header = f'<project name="{root.get("name")}" path="{root.get("path")}">\n'
        project_footer = "</project>"
        header_size = len(project_header) + len(project_footer)

        for _, files in sorted(dir_groups.items()):
            dir_content = "\n".join(files)
            dir_size = len(dir_content)

            # If adding this directory would exceed limit, start new chunk
            if current_size + dir_size + header_size > max_chunk_chars and current_chunk:
                chunk_xml = project_header + "\n".join(current_chunk) + "\n" + project_footer
                chunks.append(chunk_xml)
                current_chunk = [dir_content]
                current_size = dir_size
            else:
                current_chunk.append(dir_content)
                current_size += dir_size

        # Add final chunk
        if current_chunk:
            chunk_xml = project_header + "\n".join(current_chunk) + "\n" + project_footer
            chunks.append(chunk_xml)

        return chunks

    def _analyze_xml_chunk(
        self, chunk_xml: str, chunk_num: int, total_chunks: int, root: Path
    ) -> str:
        """Analyze a single XML chunk with orchestrator LLM."""
        if not self.llm_client:
            return "LLM not available for chunk analysis"

        chunk_context = (
            f" (analyzing part {chunk_num} of {total_chunks})" if total_chunks > 1 else ""
        )

        prompt = f"""Analyze this project structure{chunk_context} to identify issues:

**PROJECT ROOT PRINCIPLE**: A well-organized Python project keeps its root directory minimal and purposeful.

**Root directory should ONLY contain**:
- Project metadata: setup.py, pyproject.toml, MANIFEST.in, tox.ini, LICENSE
- Documentation: README.md (single entry point)
- Configuration: .gitignore, .env.example
- Package entry points: __init__.py, __main__.py, conftest.py (for testing)

**Everything else belongs in subdirectories**:
- Source code → src/ or package_name/
- Scripts and utilities → scripts/ or tools/
- Documentation → docs/
- Tests → tests/
- Examples → examples/

1. **Misplaced files**: Files in wrong directories based on their purpose
   - **Scan file paths**: Files at root level have NO `/` in their path attribute
   - **Evaluate purpose from summary**: Does this file's functionality belong at project root?
   - Common misplacements:
     * Scripts that transform/convert/modify files → should be in scripts/ or tools/
     * Utility code that's imported by the project → should be in src/
     * Additional documentation → should be in docs/
     * Test helpers or fixtures → should be in tests/

2. **Useless files**: Dead code, unused files, or files with no clear purpose
   - One-time migration scripts that have completed their purpose
   - Backup files (*.bak, *_old.py, *_backup.py)
   - Duplicate files with similar names
   - Files that are never imported or executed

3. **Redundant files**: Files with duplicate or overlapping functionality
   - Multiple files with similar summaries or purposes
   - Duplicate utility functions across modules
   - Multiple documentation files covering the same topic

4. **Overly large files**: Files >25000 bytes indicate poor modularity (check size attribute)
   - Files that could be split into logical components
   - Orchestrators or workflows that should be decomposed

5. **Consolidation opportunities**: Files that could be merged or deduplicated
   - Similar validators or workflows
   - Related documentation that should be combined

**ANALYSIS APPROACH**:
1. First, scan all files with path containing no `/` → these are at root level
2. For each root-level file, evaluate: "Should this be at root based on the principles above?"
3. If NO, flag it as misplaced with its actual purpose and recommended location
4. Then analyze the rest of the structure for other categories

Project: {root.name}

{chunk_xml}

Provide specific findings for each category with file paths and consolidation recommendations:"""

        try:
            request = self.LLMRequest(
                content=prompt, max_tokens=self.orchestrator_max_tokens, temperature=0.2
            )

            response = self.llm_client.process_request_sync(request)
            if response and response.success and response.content:
                return response.content.strip()
            else:
                return f"Chunk {chunk_num} analysis failed"

        except Exception as e:
            logger.error(f"Chunk {chunk_num} analysis failed: {e}")
            return f"Chunk {chunk_num} error: {e}"

    def _build_tree_structure(self, files: List[Path], project_root: Path) -> str:
        """Build a tree-style representation of the file structure (paths only)."""
        tree_lines = []

        for file_path in sorted(files):
            rel_path = file_path.resolve().relative_to(project_root)
            tree_lines.append(str(rel_path))

        return "\n".join(tree_lines)

    def _load_project_rules(self, project_root: Path) -> str:
        """Load project-specific justification rules from AGENTS.instructions.md."""
        agents_file = project_root / "AGENTS.instructions.md"

        if not agents_file.exists():
            return ""

        try:
            content = agents_file.read_text()

            # Extract the "File Organization & Project Structure Rules" section
            if "## File Organization & Project Structure Rules" in content:
                # Find the section
                start = content.find("## File Organization & Project Structure Rules")
                # Find the next ## heading or end of file
                next_section = content.find("\n## ", start + 1)

                if next_section == -1:
                    rules_section = content[start:]
                else:
                    rules_section = content[start:next_section]

                return rules_section.strip()

            return ""
        except Exception as e:
            logger.warning(f"Failed to load project rules from AGENTS.instructions.md: {e}")
            return ""

    def _analyze_structure(self, files: List[Path], project_root: Path) -> str:
        """Phase 1: Analyze project structure based on file paths alone."""
        if not self.llm_client:
            return "LLM not available for structural analysis"

        tree = self._build_tree_structure(files, project_root)
        project_rules = self._load_project_rules(project_root)

        if project_rules:
            self._log("Loaded project-specific rules from AGENTS.instructions.md")

        # Build base prompt
        base_prompt = """Analyze this project's file structure for organizational issues.

**PROJECT ROOT PRINCIPLE**: A well-organized Python project keeps its root directory minimal.

**Root directory should ONLY contain**:
- Project metadata: setup.py, pyproject.toml, MANIFEST.in, tox.ini, LICENSE
- Single documentation entry point: README.md
- Configuration: .gitignore, .env.example
- Package entry points: __init__.py, __main__.py, conftest.py
- AI/Agent instructions: CLAUDE.md, AGENTS.instructions.md, *.instructions.md

**Everything else belongs in subdirectories**:
- Source code → src/ or package_name/
- Documentation → docs/
- Tests → tests/
- Examples → examples/"""

        # Add project-specific rules if found
        if project_rules:
            prompt = f"""{base_prompt}

**PROJECT-SPECIFIC RULES** (from AGENTS.instructions.md):
{project_rules}

**File Structure** ({len(files)} files):
```
{tree}
```

**Task**: Identify files that are misplaced based on BOTH the general principles above AND the project-specific rules. For each misplaced file:
1. State the file path
2. Explain why it's misplaced (what principle or project rule it violates)
3. Suggest where it should be moved OR if it should be deleted

Focus on:
- Root-level files that don't belong there
- Files in wrong subdirectories
- Forbidden directories (per project rules)
- One-off scripts that should be deleted or integrated

Be specific and direct. List misplaced files clearly."""
        else:
            prompt = f"""{base_prompt}

**File Structure** ({len(files)} files):
```
{tree}
```

**Task**: Identify files that are misplaced based ONLY on their location. For each misplaced file:
1. State the file path
2. Explain why it's misplaced (what principle it violates)
3. Suggest where it should be moved

Focus on:
- Root-level files that don't belong there
- Files in wrong subdirectories
- Missing organizational structure

Be specific and direct. List misplaced files clearly."""

        try:
            self._log("Running structural analysis (file paths only)...")
            request = self.LLMRequest(
                content=prompt, max_tokens=self.orchestrator_max_tokens, temperature=0.2
            )

            response = self.llm_client.process_request_sync(request)
            if response and response.success and response.content:
                return response.content.strip()
            else:
                return "Structural analysis failed"

        except Exception as e:
            logger.error(f"Structural analysis failed: {e}")
            return f"Structural analysis error: {e}"

    def _synthesize_chunk_analyses(
        self, structural_analysis: str, chunk_analyses: List[str], root: Path
    ) -> str:
        """Synthesize structural + semantic analyses into final report."""
        if not self.llm_client:
            return f"## Structural Analysis\n\n{structural_analysis}\n\n" + "\n\n---\n\n".join(
                chunk_analyses
            )

        combined_semantic = "\n\n---\n\n".join(
            [
                f"**Semantic Analysis Part {i+1}:**\n{analysis}"
                for i, analysis in enumerate(chunk_analyses)
            ]
        )

        prompt = f"""Synthesize this multi-phase project analysis into a comprehensive report:

**PHASE 1: Structural Analysis (file organization)**
{structural_analysis}

**PHASE 2: Semantic Analysis (file content and purpose)**
{combined_semantic}

Create a unified report that:
1. Combines structural and semantic findings (avoid duplication)
2. Cross-references findings from both phases
3. Provides prioritized, actionable recommendations
4. Groups related issues together

**CRITICAL**: Do NOT include time estimates, timelines, or effort predictions (e.g., "1 week", "4-6 weeks", "15-30 minutes"). Focus on WHAT should be done and WHY, not HOW LONG it will take. The developer will determine timing based on their context.

Project: {root.name}"""

        try:
            request = self.LLMRequest(
                content=prompt, max_tokens=self.orchestrator_max_tokens, temperature=0.2
            )

            response = self.llm_client.process_request_sync(request)
            if response and response.success and response.content:
                return response.content.strip()
            else:
                # Fallback to simple concatenation
                return (
                    f"## Structural Analysis\n\n{structural_analysis}\n\n"
                    + f"## Semantic Analysis\n\n{combined_semantic}"
                )

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return (
                f"## Structural Analysis\n\n{structural_analysis}\n\n"
                + f"## Semantic Analysis\n\n{combined_semantic}"
            )

    def run_justification(self, project_root: Path) -> Dict:
        """Run the complete justification workflow."""
        try:
            start_time = time.time()
            # Ensure project_root is absolute
            project_root = project_root.resolve()
            self._log(f"Starting justification analysis for: {project_root}")

            # Step 1: Discover files
            files = self._discover_files(project_root)
            self._log(f"Discovered {len(files)} files")

            # Step 2: Initialize XML file for streaming
            self.xml_output.parent.mkdir(parents=True, exist_ok=True)
            with open(self.xml_output, "w") as xml_file:
                xml_file.write(f'<project name="{project_root.name}" path="{project_root}">\n')

                # Step 3: Summarize files and write to XML incrementally
                cache_hits = 0
                cache_misses = 0

                for idx, file_path in enumerate(files, 1):
                    file_hash = self._get_file_hash(file_path)
                    # Ensure file_path is absolute before making relative
                    rel_path = file_path.resolve().relative_to(project_root)

                    # Get or generate summary
                    if file_hash in self.summary_cache:
                        summary = self.summary_cache[file_hash]
                        cache_hits += 1
                        print(f"[{idx}/{len(files)}] Cached: {rel_path}")
                    else:
                        print(f"[{idx}/{len(files)}] Summarizing: {rel_path}")
                        self._log(f"Summarizing: {rel_path}")
                        summary = self._summarize_file(file_path)
                        self.summary_cache[file_hash] = summary
                        cache_misses += 1
                        # Save cache after each new summary
                        self._save_cache()

                    # Write file entry to XML immediately with CDATA wrapping
                    xml_file.write(
                        f'  <file path="{rel_path}" size="{file_path.stat().st_size}">\n'
                    )
                    xml_file.write(f"    <summary><![CDATA[{summary}]]></summary>\n")
                    xml_file.write("  </file>\n")
                    xml_file.flush()  # Force write to disk

                # Close project tag
                xml_file.write("</project>\n")

            self._log(f"File summaries: {cache_hits} cached, {cache_misses} new")
            self._log(f"XML written to: {self.xml_output}")

            # Step 4: PHASE 1 - Structural analysis (file paths only)
            self._log("=" * 60)
            self._log("PHASE 1: Structural Analysis (file organization)")
            self._log("=" * 60)
            structural_analysis = self._analyze_structure(files, project_root)

            # Step 5: PHASE 2 - Semantic analysis (file content/purpose)
            self._log("=" * 60)
            self._log("PHASE 2: Semantic Analysis (file content)")
            self._log("=" * 60)

            xml_content = self.xml_output.read_text()
            xml_size_chars = len(xml_content)
            xml_size_tokens = xml_size_chars // 3

            self._log(f"XML size: {xml_size_chars:,} chars (~{xml_size_tokens:,} tokens)")

            # Chunk XML semantically by directory structure
            xml_chunks = self._chunk_xml_semantically(xml_content)
            self._log(f"XML chunked into {len(xml_chunks)} parts for semantic analysis")

            # Analyze each chunk
            chunk_analyses = []
            for i, chunk in enumerate(xml_chunks):
                self._log(f"Analyzing semantic chunk {i+1}/{len(xml_chunks)}...")
                chunk_analysis = self._analyze_xml_chunk(
                    chunk, i + 1, len(xml_chunks), project_root
                )
                chunk_analyses.append(chunk_analysis)

            # Step 6: Synthesize both phases into final report
            self._log("=" * 60)
            self._log("SYNTHESIS: Combining structural + semantic findings")
            self._log("=" * 60)
            analysis = self._synthesize_chunk_analyses(
                structural_analysis, chunk_analyses, project_root
            )

            # Step 5: Generate final report
            duration = time.time() - start_time
            report = f"""# Project Justification Analysis

**Project:** {project_root.name}
**Files Analyzed:** {len(files)}
**Cache Performance:** {cache_hits} hits, {cache_misses} misses
**Analysis Time:** {duration:.1f}s

## Orchestrator Analysis

{analysis}

## Project Structure

See: {self.xml_output}
"""

            # Save report with timestamp
            report_file = Path(f".vibes/reports/justification_analysis_{self.timestamp}.md")
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text(report)

            logger.info(f"Analysis complete in {duration:.1f}s")

            return {
                "success": True,
                "exit_code": 0,
                "report": report,
                "files_analyzed": len(files),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "duration": duration,
            }

        except Exception as e:
            logger.error(f"Justification analysis failed: {e}")
            return {
                "success": False,
                "exit_code": 1,
                "error": str(e),
                "report": f"Analysis failed: {e}",
            }

    # BaseWorkflow abstract method implementations
    def execute(self, project_root: Path, context: dict) -> "WorkflowResult":
        """Execute justification analysis workflow."""
        from vibelint.workflows.core.base import WorkflowMetrics, WorkflowResult, WorkflowStatus

        start_time = time.time()

        try:
            result = self.run_justification(project_root)
            end_time = time.time()

            metrics = WorkflowMetrics(
                start_time=start_time,
                end_time=end_time,
                files_processed=result.get("files_analyzed", 0),
                custom_metrics={
                    "cache_hits": result.get("cache_hits", 0),
                    "cache_misses": result.get("cache_misses", 0),
                },
            )
            metrics.finalize()

            return WorkflowResult(
                workflow_id=self.workflow_id,
                status=WorkflowStatus.COMPLETED if result.get("success") else WorkflowStatus.FAILED,
                metrics=metrics,
                artifacts={"report": result.get("report", "")},
                error_message=result.get("error"),
            )
        except Exception as e:
            end_time = time.time()
            metrics = WorkflowMetrics(start_time=start_time, end_time=end_time)
            metrics.finalize()

            return WorkflowResult(
                workflow_id=self.workflow_id,
                status=WorkflowStatus.FAILED,
                metrics=metrics,
                error_message=str(e),
            )

    def get_required_inputs(self) -> set:
        """No required inputs."""
        return set()

    def get_produced_outputs(self) -> set:
        """Produces justification analysis outputs."""
        return {"justification_report", "xml_structure"}

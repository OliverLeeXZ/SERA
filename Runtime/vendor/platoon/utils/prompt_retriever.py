"""Prompt template retriever for Jinja templates."""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template, TemplateNotFound


class PromptRetriever:
    """Retrieves and resolves Jinja prompt templates from the prompts folder."""

    def __init__(self, prompts_dir: str | Path | None = None):
        """Initialize the prompt retriever.

        Args:
            prompts_dir: Path to the prompts directory. If None, uses default location
                        relative to this module.
        """
        if prompts_dir is None:
            # Default to prompts folder relative to this module
            current_dir = Path(__file__).parent
            prompts_dir = current_dir.parent / "prompts"

        self.prompts_dir = Path(prompts_dir)

        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")

        # Initialize Jinja environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.prompts_dir)),
            autoescape=False,  # Disable autoescaping for prompt templates
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,  # Raise error on undefined variables
        )

        # Cache for loaded templates
        self._template_cache: dict[str, Template] = {}

    def get_template_names(self) -> list[str]:
        """Get list of available template names (without .jinja extension).

        Returns:
            List of template names available in the prompts directory.
        """
        template_files = []
        for file_path in self.prompts_dir.glob("*.jinja"):
            template_files.append(file_path.stem)
        return sorted(template_files)

    def template_exists(self, template_name: str) -> bool:
        """Check if a template exists.

        Args:
            template_name: Name of the template (without .jinja extension).

        Returns:
            True if template exists, False otherwise.
        """
        template_path = self.prompts_dir / f"{template_name}.jinja"
        return template_path.exists()

    def get_template(self, template_name: str) -> Template:
        """Get a Jinja template by name.

        Args:
            template_name: Name of the template (without .jinja extension).

        Returns:
            Jinja Template object.

        Raises:
            TemplateNotFound: If template doesn't exist.
        """
        if template_name not in self._template_cache:
            try:
                self._template_cache[template_name] = self.env.get_template(f"{template_name}.jinja")
            except TemplateNotFound:
                raise TemplateNotFound(f"Template '{template_name}' not found in {self.prompts_dir}")

        return self._template_cache[template_name]

    def render_template(self, template_name: str, **variables: Any) -> str:
        """Render a template with the given variables.

        Args:
            template_name: Name of the template (without .jinja extension).
            **variables: Variables to pass to the template.

        Returns:
            Rendered template string.

        Raises:
            TemplateNotFound: If template doesn't exist.
        """
        template = self.get_template(template_name)
        return str(template.render(**variables))

    def get_prompt(self, prompt_name: str, **variables: Any) -> str:
        """Get a prompt string by name, rendering the template with variables.

        This is a convenience method that combines template retrieval and rendering.

        Args:
            prompt_name: Name of the prompt template (without .jinja extension).
            **variables: Variables to pass to the template.

        Returns:
            Rendered prompt string.

        Raises:
            TemplateNotFound: If template doesn't exist.
        """
        return self.render_template(prompt_name, **variables)

    def get_raw_template_content(self, template_name: str) -> str:
        """Get the raw content of a template file without rendering.

        Args:
            template_name: Name of the template (without .jinja extension).

        Returns:
            Raw template content as string.

        Raises:
            FileNotFoundError: If template file doesn't exist.
        """
        template_path = self.prompts_dir / f"{template_name}.jinja"
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

        return template_path.read_text(encoding="utf-8")

    def list_prompts(self) -> dict[str, str]:
        """Get a dictionary mapping prompt names to their raw content.

        Returns:
            Dictionary with prompt names as keys and raw content as values.
        """
        prompts = {}
        for template_name in self.get_template_names():
            prompts[template_name] = self.get_raw_template_content(template_name)
        return prompts


# Convenience function for quick access
def get_prompt(prompt_name: str, prompts_dir: str | Path | None = None, **variables: Any) -> str:
    """Quick function to get a prompt string by name.

    Args:
        prompt_name: Name of the prompt template (without .jinja extension).
        prompts_dir: Path to the prompts directory. If None, uses default location
                        relative to this module.
        **variables: Variables to pass to the template.

    Returns:
        Rendered prompt string.
    """
    retriever = PromptRetriever(prompts_dir=prompts_dir)
    return retriever.get_prompt(prompt_name, **variables)

"""
Generate pipeline figures using OpenAI's image generation API (DALL-E 3 or GPT-4o).

Usage:
    python scripts/generate_figure.py \
        --prompt "A clean academic pipeline diagram showing..." \
        --output paper/figures/pipeline.png \
        --size 1792x1024 \
        --model dall-e-3

    # Or use a prompt file:
    python scripts/generate_figure.py \
        --prompt_file paper/figures/pipeline_prompt.txt \
        --output paper/figures/pipeline.png

Requires: OPENAI_API_KEY in .claude/commands/.env or environment variable.
"""

import argparse
import os
import sys
import base64
from pathlib import Path

# .env lives with the skills at .claude/commands/.env
ENV_PATHS = [
    Path(__file__).parent.parent / ".claude" / "commands" / ".env",  # primary
    Path(__file__).parent.parent / ".env",                            # fallback
]


def load_env():
    """Load .env file from .claude/commands/.env (or fallback to project root)."""
    for env_path in ENV_PATHS:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())
            break


def generate_image(prompt, output_path, model="dall-e-3", size="1792x1024",
                   quality="hd", style="natural"):
    """Generate an image using OpenAI API and save to output_path."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found.")
        print("  Set it in .claude/commands/.env or as an environment variable.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Prepend academic figure style instructions
    full_prompt = (
        "Create a clean, professional academic paper figure with white background. "
        "Use a modern, minimalist style suitable for a top-tier ML conference paper "
        "(NeurIPS/ICML/ICLR). Use blue and orange as primary colors. "
        "Include clear labels and arrows. No decorative elements. "
        f"\n\nFigure description:\n{prompt}"
    )

    print(f"Generating figure with {model}...")
    print(f"  Size: {size}")
    print(f"  Quality: {quality}")
    print(f"  Output: {output_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if model == "dall-e-3":
        response = client.images.generate(
            model="dall-e-3",
            prompt=full_prompt,
            n=1,
            size=size,
            quality=quality,
            style=style,
            response_format="b64_json",
        )
        image_data = base64.b64decode(response.data[0].b64_json)
        with open(output_path, "wb") as f:
            f.write(image_data)
        revised_prompt = response.data[0].revised_prompt
        print(f"Saved to {output_path}")
        if revised_prompt:
            print(f"Revised prompt: {revised_prompt}")

    elif model in ("gpt-4o", "gpt-4o-mini", "gpt-image-1"):
        response = client.images.generate(
            model="gpt-image-1",
            prompt=full_prompt,
            n=1,
            size=size,
        )
        # gpt-image-1 returns b64_json by default
        image_data = base64.b64decode(response.data[0].b64_json)
        with open(output_path, "wb") as f:
            f.write(image_data)
        print(f"Saved to {output_path}")

    else:
        print(f"Error: Unknown model {model}. Use 'dall-e-3', 'gpt-4o', or 'gpt-4o-mini'.")
        sys.exit(1)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate figures using OpenAI image API")
    parser.add_argument("--prompt", type=str, help="Image generation prompt")
    parser.add_argument("--prompt_file", type=str, help="Path to file containing the prompt")
    parser.add_argument("--output", type=str, required=True, help="Output image path (.png)")
    parser.add_argument("--model", type=str, default="dall-e-3",
                        choices=["dall-e-3", "gpt-4o", "gpt-4o-mini"],
                        help="Model to use for generation")
    parser.add_argument("--size", type=str, default="1792x1024",
                        help="Image size (dall-e-3: 1024x1024, 1792x1024, 1024x1792)")
    parser.add_argument("--quality", type=str, default="hd",
                        choices=["standard", "hd"],
                        help="Image quality (dall-e-3 only)")
    parser.add_argument("--style", type=str, default="natural",
                        choices=["natural", "vivid"],
                        help="Image style (dall-e-3 only)")
    args = parser.parse_args()

    load_env()

    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        print("Error: Provide --prompt or --prompt_file")
        sys.exit(1)

    generate_image(prompt, args.output, model=args.model, size=args.size,
                   quality=args.quality, style=args.style)


if __name__ == "__main__":
    main()

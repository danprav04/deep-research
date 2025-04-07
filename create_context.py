import os
import fnmatch
import pathspec # For parsing .gitignore
from pathlib import Path

# --- Configuration ---
ROOT_DIR = Path('.') # Use current directory as root
OUTPUT_FILE = "project_context.txt"
# Add patterns for files/dirs to always exclude, besides .gitignore
DEFAULT_EXCLUDES = [
    ".git/",
    ".gitattributes",
    "__pycache__/",
    "*.pyc",
    ".DS_Store",
    ".env",
    "venv/",
    "node_modules/",
    OUTPUT_FILE # Exclude the output file itself
]
# --- End Configuration ---

def get_gitignore_patterns(root_dir):
    """Reads .gitignore and returns pathspec object."""
    gitignore_path = root_dir / '.gitignore'
    patterns = []
    if gitignore_path.is_file():
        try:
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                patterns = f.read().splitlines()
        except Exception as e:
            print(f"Warning: Could not read .gitignore: {e}")

    # Combine default excludes with .gitignore content
    all_patterns = DEFAULT_EXCLUDES + patterns
    # Use GitWildMatchPattern for standard gitignore syntax
    return pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, all_patterns)

def generate_tree(start_path, spec, prefix=""):
    """Generates a directory tree string, respecting ignores."""
    tree_str = ""
    # Use scandir for potentially better performance
    try:
        entries = sorted(
            [entry for entry in os.scandir(start_path) if entry.name != '.git'], # Basic .git exclude here too
            key=lambda entry: (not entry.is_dir(), entry.name.lower())
        )
    except FileNotFoundError:
        return "" # Directory might have been deleted mid-scan

    for i, entry in enumerate(entries):
        is_last = (i == len(entries) - 1)
        connector = "└── " if is_last else "├── "
        entry_rel_path = Path(entry.path).relative_to(ROOT_DIR)

        # Check if the entry (file or dir) should be ignored
        if spec.match_file(str(entry_rel_path)) or spec.match_file(str(entry_rel_path) + '/'): # Check both file and dir patterns
             # Also check if the containing directory is ignored explicitly (needed for pathspec behavior)
            if any(spec.match_file(p) for p in entry_rel_path.parents if p != Path('.')):
                continue # Skip if parent dir is ignored
            if not entry.is_dir(): # Skip ignored files
                continue
            # For directories, we need to check if the *directory itself* matches a pattern.
            # If it does, skip it entirely. Pathspec often needs the trailing / for dirs.
            if spec.match_file(str(entry_rel_path) + '/'):
                continue

        tree_str += prefix + connector + entry.name + ("/" if entry.is_dir() else "") + "\n"

        if entry.is_dir():
            new_prefix = prefix + ("    " if is_last else "│   ")
            # Recursively generate tree for subdirectories
            subtree = generate_tree(Path(entry.path), spec, new_prefix)
            tree_str += subtree

    return tree_str

def get_file_contents(root_dir, spec):
    """Gets contents of all non-ignored files."""
    all_content = []
    for root, dirs, files in os.walk(root_dir, topdown=True):
        root_path = Path(root)
        rel_root_path_str = str(root_path.relative_to(root_dir))
        if rel_root_path_str == '.':
             rel_root_path_str = '' # Adjust for root directory itself

        # --- Directory Exclusion ---
        # Filter dirs in-place so os.walk doesn't descend into them
        original_dirs = list(dirs) # Copy since we modify dirs
        dirs[:] = [] # Clear the list, we will re-add non-ignored ones
        for d in original_dirs:
            dir_rel_path = Path(rel_root_path_str) / d if rel_root_path_str else Path(d)
            # Check if the directory path itself or with a trailing slash is ignored
            if not spec.match_file(str(dir_rel_path)) and not spec.match_file(str(dir_rel_path) + '/'):
                 dirs.append(d) # Keep this directory

        # --- File Processing ---
        for filename in files:
            file_path = root_path / filename
            rel_path = file_path.relative_to(root_dir)

            # Check if the file should be ignored
            if spec.match_file(str(rel_path)):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                all_content.append(f"--- File: {rel_path} ---\n\n```\n{content}\n```\n\n")
                print(f"Processed: {rel_path}")
            except Exception as e:
                all_content.append(f"--- File: {rel_path} --- \n\n```\nError reading file: {e}\n```\n\n")
                print(f"Error reading {rel_path}: {e}")

    return "".join(all_content)

def main():
    print("Starting context consolidation...")

    # 1. Get ignore patterns
    print("Reading .gitignore and default excludes...")
    spec = get_gitignore_patterns(ROOT_DIR)

    # 2. Generate file tree
    print("Generating file tree...")
    # Generate tree starting from the specified root directory
    project_tree = f"{ROOT_DIR.name}/\n" # Add root dir name
    project_tree += generate_tree(ROOT_DIR, spec)
    print("Tree generation complete.")

    # 3. Get file contents
    print("Reading file contents...")
    file_contents = get_file_contents(ROOT_DIR, spec)
    print("File reading complete.")

    # 4. Combine and write output
    print(f"Writing output to {OUTPUT_FILE}...")
    final_output = f"Project Root: {ROOT_DIR.resolve()}\n\n"
    final_output += "--- File Tree ---\n\n"
    final_output += "```\n"
    final_output += project_tree
    final_output += "```\n\n"
    final_output += "--- File Contents ---\n\n"
    final_output += file_contents

    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(final_output)
        print(f"Successfully created {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    main()

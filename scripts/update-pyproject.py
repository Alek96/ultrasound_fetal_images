import re
import subprocess

import toml


def get_poetry_top_level():
    result = subprocess.run(["poetry", "show", "--top-level"], stdout=subprocess.PIPE, text=True, check=True)
    packages = {}
    prog = re.compile(r"^(\S+)\s+(\S+)\s+")
    for line in result.stdout.splitlines():
        match = prog.match(line)
        if match:
            name, version = match.groups()
            packages[name] = version
    return packages


def update_pyproject_toml_text(packages, toml_path="pyproject.toml"):
    with open(toml_path) as f:
        lines = f.readlines()

    dep_section = False
    new_lines = []
    for line in lines:

        if dep_section:
            if line.strip().startswith("["):
                dep_section = False
            else:
                line = replace_version(line, packages)
        elif line.strip().startswith("[tool.poetry.dependencies]"):
            dep_section = True

        new_lines.append(line)

    # for line in new_lines:
    #     print(line, end="")
    with open(toml_path, "w") as f:
        f.writelines(new_lines)
    print("pyproject.toml updated.")


def replace_version(line, packages):
    match = re.match(r'^(\S+)\s+=\s+"(\^)?(\S+)"', line)
    if match:
        name, up_sign, old_version = match.groups()
        _, old_version_end = match.span(3)

        if name in packages:
            new_version = packages[name]
            # Replace only if version differs
            if new_version != old_version:
                line = f'{name} = "{up_sign}{new_version}{line[old_version_end:]}'
    return line


if __name__ == "__main__":
    pkgs = get_poetry_top_level()
    update_pyproject_toml_text(pkgs)

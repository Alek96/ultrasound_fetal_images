import re
from pathlib import Path

from poetry.factory import Factory


def get_installed_top_level_versions(pyproject_path="pyproject.toml"):
    poetry = Factory().create_poetry(Path(pyproject_path))
    top_level = {dep.name for dep in poetry.package.requires}
    lock_data = poetry.locker.lock_data
    installed = {}
    for pkg in lock_data["package"]:
        name = pkg["name"]
        if name in top_level:
            installed[name] = pkg["version"]
    return installed


def update_pyproject(packages, pyproject_path="pyproject.toml"):
    with open(pyproject_path) as f:
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
    with open(pyproject_path, "w") as f:
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
    pkgs = get_installed_top_level_versions()
    update_pyproject(pkgs)

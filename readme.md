# Overview

Particula-Beta is the development platform for the Particula package.



---

## Installation

You can install `particula-beta` directly from GitHub using `pip`. This allows you to get the latest version of the package, even if it is not yet published on PyPI.

### Install the Latest Version

To install the latest version from the main branch, run:

```bash
pip install git+https://github.com/uncscode/particula-beta.git
```

### Install a Specific Branch, Tag, or Commit

If you want to install a specific version of `particula-beta`, you can specify a branch, tag, or commit in the installation command.

- **Install from a specific branch** (e.g., `dev` branch):
  ```bash
  pip install git+https://github.com/uncscode/particula-beta.git@dev
  ```

- **Install a specific release version** (e.g., `v0.1.0`):
  ```bash
  pip install git+https://github.com/uncscode/particula-beta.git@v0.1.0
  ```

- **Install from a specific commit** (replace `commit_hash` with an actual commit hash):
  ```bash
  pip install git+https://github.com/uncscode/particula-beta.git@commit_hash
  ```

### Verifying Installation

After installation, you can verify that `particula-beta` is installed correctly by running:

```bash
python -c "import particula_beta; print(particula_beta.__version__)"
```

If the package was installed successfully, this command will print the installed version of `particula-beta`.

---

## Upgrading `particula-beta`

To upgrade to the latest version of `particula-beta`, use:

```bash
pip install --upgrade git+https://github.com/uncscode/particula-beta.git
```

or

```bash
pip install -U git+https://github.com/uncscode/particula-beta.git
```

If you need a specific branch, tag, or commit:

- Upgrade to a specific branch:
  ```bash
  pip install -U git+https://github.com/uncscode/particula-beta.git@dev
  ```
- Upgrade to a specific release version:
  ```bash
  pip install -U git+https://github.com/uncscode/particula-beta.git@v1.0.0
  ```
- Upgrade to a specific commit:
  ```bash
  pip install -U git+https://github.com/uncscode/particula-beta.git@commit_hash
  ```

If the upgrade does not apply correctly, you can force a reinstall:

```bash
pip install --force-reinstall git+https://github.com/uncscode/particula-beta.git
```

---

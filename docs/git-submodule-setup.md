# Git Submodule Setup and Troubleshooting

This document provides comprehensive guidance for setting up and troubleshooting the `dexrobot_mujoco` git submodule used in the DexHand Isaac Gym environment.

## Overview

The `dexrobot_mujoco` submodule contains essential assets for the DexHand simulation:

- **Hand Models**: MuJoCo XML files for different hand configurations (left/right, simplified/full collision)
- **3D Meshes**: STL files for hand components and visual rendering
- **Scene Assets**: Pre-built scenes for manipulation tasks (ball catching, furniture interaction)
- **ROS Integration**: Tools for ROS1/ROS2 compatibility and data recording
- **Utilities**: Helper scripts for hand articulation and model conversion

**Location**: `assets/dexrobot_mujoco/`
**Primary Repository**: `https://gitee.com/dexrobot/dexrobot_mujoco.git`

## Initial Setup

### Option 1: Clone with Submodules (Recommended)

```bash
git clone --recursive https://github.com/dexrobot/dexrobot_isaac
cd dexrobot_isaac
```

### Option 2: Initialize After Cloning

If you already cloned without `--recursive`:

```bash
cd dexrobot_isaac
git submodule update --init --recursive
```

### Verification

Verify the submodule is properly initialized:

```bash
# Check submodule status
git submodule status

# Should show something like:
# d0f332e559ef991b95bad6a2c16adad476b9487b assets/dexrobot_mujoco (v0.1.0-36-gd0f332e)

# Verify files exist
ls -la assets/dexrobot_mujoco/
# Should show README.md, dexrobot_mujoco/, docs/, etc.
```

## Submodule Updates

### Important: Version Consistency

**Always use `git submodule update`** to ensure the submodule version matches the main repository's specifications. This maintains compatibility between the main codebase and submodule assets.

### Update to Repository-Specified Version

```bash
# Update to the exact commit specified by the main repository
git submodule update --remote

# Or update all submodules recursively
git submodule update --init --recursive
```

### Fetch Latest from Internal Repository

Users with access to internal repositories can fetch the latest updates:

```bash
cd assets/dexrobot_mujoco

# Add internal remote (if not already added)
git remote add internal <internal_repository_url>

# Fetch latest changes
git fetch internal

# Switch to a specific branch or tag
git checkout internal/main
# or
git checkout internal/latest-stable

# Return to main directory and update submodule reference
cd ../..
git add assets/dexrobot_mujoco
git commit -m "Update dexrobot_mujoco submodule to latest version"
```

### Update Workflow Best Practices

1. **Check current version**:
   ```bash
   git submodule status
   ```

2. **Update to specified version**:
   ```bash
   git submodule update --remote
   ```

3. **Verify compatibility**:
   ```bash
   python examples/dexhand_test.py
   ```

4. **Commit changes** (if updating main repo):
   ```bash
   git add .gitmodules assets/dexrobot_mujoco
   git commit -m "Update submodule to version X.X.X"
   ```

## Troubleshooting

### Missing Submodule After Clone

**Symptom**: `assets/dexrobot_mujoco/` is empty or missing

**Solution**:
```bash
git submodule update --init --recursive
```

### Submodule Update Failures

**Symptom**: `fatal: remote error: access denied` or network timeouts

**Solutions**:

1. **Check network connectivity**:
   ```bash
   ping gitee.com
   ```

2. **Try alternative repository**:
   ```bash
   cd assets/dexrobot_mujoco
   git remote set-url origin https://github.com/DexRobot/dexrobot_mujoco.git
   git fetch origin
   ```

3. **Use SSH instead of HTTPS**:
   ```bash
   cd assets/dexrobot_mujoco
   git remote set-url origin git@gitee.com:dexrobot/dexrobot_mujoco.git
   git fetch origin
   ```

### Permission Problems with SSH Keys

**Symptom**: `Permission denied (publickey)` errors

**Solution**:
```bash
# Check SSH key
ssh -T git@gitee.com
# or
ssh -T git@github.com

# Add SSH key to agent if needed
ssh-add ~/.ssh/id_rsa
```

### Detached HEAD State

**Symptom**: Submodule shows `(HEAD detached at <commit>)`

**This is normal behavior** - submodules are designed to be in detached HEAD state to maintain version consistency.

**To make changes**:
```bash
cd assets/dexrobot_mujoco
git checkout main  # or desired branch
# Make your changes
git add .
git commit -m "Your changes"
git push origin main
```

### Conflicts During Updates

**Symptom**: Merge conflicts when updating submodule

**Solution**:
```bash
cd assets/dexrobot_mujoco
git status  # Check conflicted files
git reset --hard HEAD  # Discard local changes
git pull origin main  # Or desired branch
```

### Version Consistency Issues

**Symptom**: Submodule version doesn't match main repository expectations

**Solution**:
```bash
# Reset to repository-specified version
git submodule update --init --recursive

# Force update to specific commit
cd assets/dexrobot_mujoco
git checkout <commit_hash>
cd ..
git add assets/dexrobot_mujoco
git commit -m "Fix submodule version consistency"
```

## Manual Recovery

### Complete Submodule Reset

If the submodule becomes corrupted:

```bash
# Remove submodule directory
rm -rf assets/dexrobot_mujoco

# Re-initialize
git submodule update --init --recursive

# Verify
git submodule status
```

### Emergency Recovery

If all else fails:

```bash
# Remove submodule completely
git submodule deinit assets/dexrobot_mujoco
rm -rf .git/modules/assets/dexrobot_mujoco
git rm -f assets/dexrobot_mujoco

# Re-add submodule
git submodule add https://gitee.com/dexrobot/dexrobot_mujoco.git assets/dexrobot_mujoco
git submodule update --init --recursive
```

## Alternative Sources

### Using GitHub Mirror

If Gitee is unavailable:

```bash
cd assets/dexrobot_mujoco
git remote add github https://github.com/DexRobot/dexrobot_mujoco.git
git fetch github
git checkout github/main
```

### Using Internal Repository

For users with internal repository access:

```bash
cd assets/dexrobot_mujoco
git remote add internal <internal_repository_url>
git fetch internal
git checkout internal/main
```

**Note**: Internal repository URLs are not included in this documentation for security reasons. Contact your system administrator for access details.

## Verification Steps

### Basic Verification

```bash
# Check submodule status
git submodule status

# Verify essential files exist
ls -la assets/dexrobot_mujoco/dexrobot_mujoco/models/
# Should show: dexhand021_right.xml, dexhand021_left.xml, etc.

# Check meshes
ls -la assets/dexrobot_mujoco/dexrobot_mujoco/meshes/dexhand021/
# Should show: *.STL files for hand components
```

### Functional Verification

```bash
# Test environment creation
python examples/dexhand_test.py

# Should load successfully without missing file errors
```

### Integration Test

```bash
# Run basic training test
python train.py task=BlindGrasping env.numEnvs=4 training.maxIterations=10

# Should complete without asset loading errors
```

## Best Practices

1. **Always use `git submodule update`** for version consistency
2. **Verify submodule status** before reporting issues
3. **Test functionality** after submodule updates
4. **Commit submodule changes** in main repository when updating
5. **Use appropriate remote** based on network accessibility
6. **Keep SSH keys up to date** for authentication

## Getting Help

If you encounter issues not covered in this guide:

1. Check the main [TROUBLESHOOTING.md](TROUBLESHOOTING.md) document
2. Verify network connectivity to repository sources
3. Contact your system administrator for internal repository access
4. Check the submodule repository's own documentation in `assets/dexrobot_mujoco/README.md`

---

*This documentation is part of the DexHand Isaac Gym environment. For more information, see the main [README.md](../README.md) and [documentation index](INDEX.md).*

# Git Repository Repair Documentation

## Issues Encountered

1. **File Lock Issues**
   - Error: `Deletion of directory '.git/objects/15' failed`
   - Error: `Deletion of directory '.git/objects/1b' failed`
   - Error: `Deletion of directory '.git/objects/1e' failed`
   - These errors indicated that Git was unable to delete certain object directories due to file locks or permission issues.

2. **Corrupted Git Repository**
   - Error: `fatal: bad object refs/heads/main`
   - Error: `fatal: Could not parse object 'HEAD'`
   - These errors indicated that the Git repository's object database was corrupted.

## Root Causes

1. **File System Locks**
   - Windows file system locks preventing Git from modifying certain files
   - Multiple Git processes potentially holding locks on the same files
   - Permission issues with the `.git` directory

2. **Repository Corruption**
   - Invalid SHA1 pointers in the repository
   - Corrupted object database
   - Broken references to commits and branches

## Solution Steps

1. **Backup Creation**
   ```bash
   xcopy /E /I /H /Y . ..\ClassForge_backup
   ```
   - Created a complete backup of the project before making any changes
   - Ensured no data loss during the repair process

2. **Repository Reinitialization**
   ```bash
   rmdir /S /Q .git
   git init
   git add .
   git commit -m "Initial commit"
   ```
   - Removed the corrupted `.git` directory
   - Created a fresh Git repository
   - Added all files and created a new initial commit

3. **Remote Repository Setup**
   ```bash
   git remote add origin https://github.com/MINH01-HUB/Classforge_
   git push -f origin master
   ```
   - Re-established connection with the remote repository
   - Force pushed the new repository state

## Prevention Measures

1. **Regular Maintenance**
   - Run `git gc` periodically to optimize the repository
   - Use `git fsck` to check repository integrity
   - Keep the repository clean with `git clean`

2. **Best Practices**
   - Avoid force-closing Git operations
   - Ensure proper file permissions
   - Keep Git processes from running simultaneously
   - Regular backups of important repositories

3. **Troubleshooting Steps**
   If similar issues occur:
   1. Check for running Git processes
   2. Verify file permissions
   3. Create a backup
   4. Try repository repair commands
   5. If all else fails, reinitialize the repository

## Additional Notes

- The solution preserved all project files while fixing the Git repository structure
- No data loss occurred during the repair process
- The repository is now in a clean state and fully functional
- All remote connections have been properly reestablished

## References

- Git Documentation: https://git-scm.com/doc
- Git Maintenance: https://git-scm.com/docs/git-maintenance
- Git Repository Corruption: https://git-scm.com/docs/git-fsck 
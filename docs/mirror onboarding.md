# MIRROR Onboarding

## General Pipeline Information

### Initial Access Setup

 [Add Content Here]

### Things to do every time you log in

1. Log in to the supercomputer

    If you have a VSCode window already open to the MIRROR Pipeline, this will simply consist of logging in again with your password and authentication code. Otherwise, go to the Remote Explorer tab (computer monitor icon), hover over the MIRROR-Pipeline, and click the arrow icon, then log in. 

2. (If necessary) Run `source /etc/profile`

    If you logged in to the supercomputer strictly by ssh'ing through the terminal, this will automatically have been done for you. Otherwise (e.g. using VSCode to log in to the supercomputer), run `source /etc/profile/` to load system-wide environment settings and paths. 
    
3. Activate mamba environment

    To activate the project's conda environment, run `mamba activate ./.env`. (You'll have to have [created the environment first](#initial-access-setup), of course.) 

4. Ensure you're on the correct branch

    Run `git status` to see what branch you're currently checked out to. Make sure that this matches the branch you're intending to work on. If it doesn't, run `git checkout <branch_name>` to switch to the correct branch. 

5. Update with any new changes

    Run `git fetch origin` to check for any new changes to the main branch from other branches getting merged, then `git merge main` to integrate these changes into your local branch. If there are merge conflicts, resolve them as needed. 

    If you're working on a branch with someone else, run `git pull` to check for any changes they may have pushed and integrate them locally. 
 
### General MIRROR Pipeline Architecture

[the core components, data flow, and primary services that make up the MIRROR Pipeline]

[how to use the MIRROR Pipeline. This will essentially be the documentation for our pipeline]
 
## Pipeline Developer Information

### Pre-Pull Request Checklist
[specific test runs, formatting checks they must run locally before opening a Pull Request]
 
### Ticket 5: New Ticket Kickoff Checklist
[administrative steps required for assigning yourself a new ticket (move the card, assign yourself, create a branch, etc.), and also for writing a new ticket]

#!/bin/bash

# Sentiment Analysis Project Deployment Script

# This script provides instructions for deploying the sentiment analysis project to GitHub

echo "==============================================="
echo "Sentiment Analysis Project - GitHub Deployment"
echo "==============================================="

echo
echo "IMPORTANT: This script will guide you through the process of deploying"
echo "the sentiment analysis project to a new GitHub repository."
echo

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: Git is not installed. Please install Git before continuing."
    exit 1
fi

echo "Step 1: Create a new GitHub repository"
echo "-------------------------------------"
echo "1. Go to https://github.com/new"
echo "2. Enter repository name: 'sentiment-analysis-project'"
echo "3. Select 'Public' (or 'Private' if you prefer)"
echo "4. Do NOT initialize with a README, .gitignore, or license"
echo "5. Click 'Create repository'"
echo

echo "Step 2: Get your GitHub repository URL"
echo "--------------------------------------"
echo "After creating the repository, you'll see a URL that looks like:"
echo "  https://github.com/USERNAME/sentiment-analysis-project.git"
echo "or"
echo "  git@github.com:USERNAME/sentiment-analysis-project.git"
echo

read -p "Enter your GitHub repository URL: " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "Error: Repository URL is required."
    exit 1
fi

echo
echo "Step 3: Configuring Git and pushing the code"
echo "---------------------------------------------"

# Ensure we're in the right directory
cd /workspace

# Add the remote origin
git remote add origin "$REPO_URL"

# Verify the remote was added
if git remote -v | grep origin; then
    echo "Successfully added remote origin: $REPO_URL"
else
    echo "Error: Failed to add remote origin."
    exit 1
fi

echo
echo "Step 4: Pushing the code to GitHub"
echo "----------------------------------"

# Push the current branch to GitHub
git push -u origin qwen-code-483f453d-4f96-4d4a-b0ca-5b09b36ecd44

if [ $? -eq 0 ]; then
    echo
    echo "==============================================="
    echo "SUCCESS: Code has been pushed to GitHub!"
    echo "Your repository is now available at: $REPO_URL"
    echo "==============================================="
else
    echo
    echo "Error occurred during push. You may need to authenticate with GitHub."
    echo "If using HTTPS, you might need to provide your username and token."
    echo "If using SSH, ensure your SSH key is added to your GitHub account."
fi

echo
echo "Step 5: Optional - Rename the main branch to 'main'"
echo "--------------------------------------------------"
echo "After pushing, you might want to rename the branch:"
echo "  git branch -M main"
echo "  git push -u origin main"
echo
echo "Then delete the old branch from remote if desired:"
echo "  git push origin --delete qwen-code-483f453d-4f96-4d4a-b0ca-5b09b36ecd44"
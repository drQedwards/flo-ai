#!/bin/bash

# Pandora Transformer Deployment Script
# This will deploy your transformer to Railway (free hosting)

echo "🚀 Pandora Transformer Deployment Script"
echo "========================================"

# Check if transformer is compiled
if [ ! -f "./lattice_demo" ]; then
    echo "📦 Compiling transformer..."
    gcc transformer_lattice.c -lm -o lattice_demo
    if [ $? -eq 0 ]; then
        echo "✅ Transformer compiled successfully"
    else
        echo "❌ Failed to compile transformer"
        exit 1
    fi
else
    echo "✅ Transformer already compiled"
fi

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo ""
    echo "❌ Railway CLI not found. Please install it first:"
    echo ""
    echo "   npm install -g @railway/cli"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Initialize git if not already done
if [ ! -d ".git" ]; then
    echo "📝 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit - Pandora Transformer"
fi

echo ""
echo "🎯 Ready to deploy! Follow these steps:"
echo ""
echo "1. Run: railway login"
echo "2. Run: railway init"
echo "3. Run: railway up"
echo ""
echo "After deployment, your API will be available at:"
echo "https://[your-project-name].railway.app"
echo ""
echo "📚 API endpoints:"
echo "  • /docs - Interactive documentation"
echo "  • /health - Health check"
echo "  • /generate - Generate text"
echo "  • /model/info - Model information"
echo ""

# Ask if they want to proceed with deployment
read -p "Do you want to start the Railway deployment now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting Railway deployment..."
    
    # Check if already logged in
    if railway whoami &> /dev/null; then
        echo "✅ Already logged into Railway"
    else
        echo "🔐 Please login to Railway..."
        railway login
    fi
    
    # Initialize project if needed
    if [ ! -f "railway.json" ]; then
        echo "🎯 Initializing Railway project..."
        railway init
    fi
    
    # Deploy
    echo "🚀 Deploying to Railway..."
    railway up
    
    echo ""
    echo "🎉 Deployment started! Check the Railway dashboard for your URL."
    echo "💡 Your API will be available at: https://[project-name].railway.app"
    
else
    echo ""
    echo "📋 Manual deployment steps:"
    echo "1. railway login"
    echo "2. railway init"  
    echo "3. railway up"
fi

echo ""
echo "✨ Your Pandora Transformer will be live on the internet!"
echo "🌍 Share your API URL with others to let them use your model."
#!/usr/bin/env python3
"""
Grok 4 Setup Script
==================

This script sets up the complete Grok 4 environment including:
- Enhanced Transformer Lattice (C backend)
- Flo AI integration
- Model checkpoints
- Dependencies
- Configuration

Usage:
    python setup_grok4.py [options]

Options:
    --install-deps      Install Python dependencies
    --compile-lattice   Compile C transformer lattice
    --download-weights  Download Grok-1 base weights
    --setup-flo-ai      Setup Flo AI framework
    --create-config     Create configuration files
    --run-tests         Run test suite
    --all               Run all setup steps

Author: Grok 4 Development Team
License: Apache 2.0
"""

import argparse
import os
import subprocess
import sys
import urllib.request
import shutil
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Grok4Setup:
    """Setup manager for Grok 4 system"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.grok_base_dir = self.root_dir / "grok_base"
        self.logs_dir = self.root_dir / "grok4_logs"
        self.config_dir = self.root_dir / "config"
        self.models_dir = self.root_dir / "models"
        
        # Create directories
        self.logs_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
    
    def install_dependencies(self):
        """Install Python dependencies"""
        logger.info("🔧 Installing Python dependencies...")
        
        try:
            # Install requirements
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements_grok4.txt"
            ], check=True)
            
            # Install Flo AI if not available
            try:
                import flo_ai
                logger.info("✅ Flo AI already installed")
            except ImportError:
                logger.info("📦 Installing Flo AI framework...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-e", "./flo_ai"
                ], check=True)
            
            logger.info("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def compile_lattice(self):
        """Compile the enhanced transformer lattice C code"""
        logger.info("🏗️  Compiling enhanced transformer lattice...")
        
        try:
            c_file = "grok4_transformer_lattice.c"
            lib_file = "grok4_lattice.so"
            
            if not os.path.exists(c_file):
                logger.error(f"❌ C source file not found: {c_file}")
                return False
            
            # Compile with optimizations
            compile_cmd = [
                "gcc", "-shared", "-fPIC", "-O3", "-march=native",
                "-fopenmp", "-DNDEBUG", c_file, "-lm", "-o", lib_file
            ]
            
            logger.info(f"Running: {' '.join(compile_cmd)}")
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Transformer lattice compiled successfully")
                
                # Test the compiled library
                if self._test_lattice_library(lib_file):
                    logger.info("✅ Lattice library test passed")
                    return True
                else:
                    logger.warning("⚠️ Lattice library compiled but test failed")
                    return False
            else:
                logger.error(f"❌ Compilation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Compilation error: {e}")
            return False
    
    def _test_lattice_library(self, lib_file):
        """Test the compiled lattice library"""
        try:
            import ctypes
            lib = ctypes.CDLL(f"./{lib_file}")
            logger.info("Library loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Library test failed: {e}")
            return False
    
    def download_weights(self):
        """Download and setup Grok-1 base weights"""
        logger.info("📥 Setting up Grok-1 base weights...")
        
        try:
            # Check if Grok base directory exists
            if not self.grok_base_dir.exists():
                logger.error("❌ Grok base directory not found. Please clone grok-1 first.")
                return False
            
            checkpoints_dir = self.grok_base_dir / "checkpoints"
            
            if not checkpoints_dir.exists():
                logger.info("Creating checkpoints directory...")
                checkpoints_dir.mkdir(exist_ok=True)
                
                # Create placeholder checkpoint info
                checkpoint_info = {
                    "model_type": "grok-1",
                    "version": "1.0",
                    "parameters": "314B",
                    "notes": "Placeholder for actual Grok-1 weights"
                }
                
                with open(checkpoints_dir / "checkpoint_info.json", "w") as f:
                    json.dump(checkpoint_info, f, indent=2)
                
                logger.info("📝 Created placeholder checkpoint directory")
                logger.info("ℹ️  To use actual Grok-1 weights, follow the download instructions in the Grok-1 repository")
            else:
                logger.info("✅ Checkpoints directory already exists")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup weights: {e}")
            return False
    
    def setup_flo_ai(self):
        """Setup and configure Flo AI framework"""
        logger.info("🌊 Setting up Flo AI framework...")
        
        try:
            # Check Flo AI installation
            import flo_ai
            logger.info(f"✅ Flo AI version: {getattr(flo_ai, '__version__', 'unknown')}")
            
            # Create sample Flo AI configuration
            flo_config = {
                "session": {
                    "default_model": "grok-4",
                    "temperature": 0.7,
                    "max_tokens": 4096
                },
                "agents": {
                    "default_tools": ["Grok4Reasoning", "Grok4WebSearch", "Grok4CodeGeneration"],
                    "logging": True
                },
                "teams": {
                    "enable_supervision": True,
                    "max_concurrent_agents": 5
                }
            }
            
            config_file = self.config_dir / "flo_ai_config.json"
            with open(config_file, "w") as f:
                json.dump(flo_config, f, indent=2)
            
            logger.info(f"✅ Flo AI configuration saved to {config_file}")
            return True
            
        except ImportError:
            logger.error("❌ Flo AI not installed. Run --install-deps first.")
            return False
        except Exception as e:
            logger.error(f"❌ Flo AI setup failed: {e}")
            return False
    
    def create_config(self):
        """Create configuration files"""
        logger.info("⚙️  Creating configuration files...")
        
        try:
            # Main Grok 4 configuration
            grok4_config = {
                "model": {
                    "d_model": 6144,
                    "n_heads": 48,
                    "n_kv_heads": 8,
                    "n_layers": 64,
                    "vocab_size": 131072,
                    "max_seq_len": 8192
                },
                "moe": {
                    "n_experts": 8,
                    "n_experts_per_token": 2,
                    "moe_layers": list(range(1, 64, 2))
                },
                "features": {
                    "use_rotary_emb": True,
                    "use_rms_norm": True,
                    "use_glu": True,
                    "use_kv_cache": True
                },
                "training": {
                    "learning_rate": 1e-4,
                    "warmup_steps": 10000,
                    "max_grad_norm": 1.0
                },
                "inference": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50
                },
                "flo_ai": {
                    "use_flo_ai": True,
                    "enable_reasoning": True,
                    "enable_web_search": True,
                    "enable_code_execution": True
                }
            }
            
            config_file = self.config_dir / "grok4_config.json"
            with open(config_file, "w") as f:
                json.dump(grok4_config, f, indent=2)
            
            # Environment configuration
            env_config = """# Grok 4 Environment Configuration
# ================================

# Model paths
GROK4_MODEL_PATH=./models
GROK4_CHECKPOINT_PATH=./grok_base/checkpoints
GROK4_CONFIG_PATH=./config

# Logging
GROK4_LOG_LEVEL=INFO
GROK4_LOG_PATH=./grok4_logs

# Performance
GROK4_USE_GPU=true
GROK4_MAX_MEMORY_GB=32
GROK4_NUM_WORKERS=4

# API Configuration (optional)
# OPENAI_API_KEY=your_openai_key_here
# GROK_API_KEY=your_grok_key_here

# Flo AI Configuration
FLO_AI_LOG_LEVEL=INFO
FLO_AI_CACHE_DIR=./flo_ai_cache
"""
            
            env_file = self.root_dir / ".env.example"
            with open(env_file, "w") as f:
                f.write(env_config)
            
            logger.info(f"✅ Configuration files created:")
            logger.info(f"   - {config_file}")
            logger.info(f"   - {env_file}")
            logger.info("   - Copy .env.example to .env and customize as needed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create configuration: {e}")
            return False
    
    def run_tests(self):
        """Run basic tests to verify installation"""
        logger.info("🧪 Running tests...")
        
        tests_passed = 0
        total_tests = 0
        
        # Test 1: Python imports
        total_tests += 1
        try:
            import torch
            import jax
            import numpy as np
            logger.info("✅ Test 1: Core dependencies import successfully")
            tests_passed += 1
        except ImportError as e:
            logger.error(f"❌ Test 1: Import failed: {e}")
        
        # Test 2: Flo AI import
        total_tests += 1
        try:
            from flo_ai import Flo, FloSession, FloAgent
            logger.info("✅ Test 2: Flo AI imports successfully")
            tests_passed += 1
        except ImportError as e:
            logger.error(f"❌ Test 2: Flo AI import failed: {e}")
        
        # Test 3: Lattice library
        total_tests += 1
        if os.path.exists("grok4_lattice.so"):
            if self._test_lattice_library("grok4_lattice.so"):
                logger.info("✅ Test 3: Lattice library loads successfully")
                tests_passed += 1
            else:
                logger.error("❌ Test 3: Lattice library failed to load")
        else:
            logger.warning("⚠️ Test 3: Lattice library not found (compile with --compile-lattice)")
        
        # Test 4: Configuration files
        total_tests += 1
        config_file = self.config_dir / "grok4_config.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    json.load(f)
                logger.info("✅ Test 4: Configuration file is valid")
                tests_passed += 1
            except json.JSONDecodeError:
                logger.error("❌ Test 4: Configuration file is invalid")
        else:
            logger.warning("⚠️ Test 4: Configuration file not found (create with --create-config)")
        
        # Test 5: Basic Grok 4 instantiation
        total_tests += 1
        try:
            sys.path.append(str(self.root_dir))
            from grok4_flo_ai import Grok4Config, Grok4Core
            
            config = Grok4Config(d_model=128, n_heads=4, n_layers=2)  # Small config for test
            model = Grok4Core(config)
            logger.info("✅ Test 5: Grok 4 model instantiation successful")
            tests_passed += 1
        except Exception as e:
            logger.error(f"❌ Test 5: Grok 4 instantiation failed: {e}")
        
        # Summary
        logger.info(f"\n📊 Test Results: {tests_passed}/{total_tests} passed")
        
        if tests_passed == total_tests:
            logger.info("🎉 All tests passed! Grok 4 is ready to use.")
            return True
        else:
            logger.warning(f"⚠️ {total_tests - tests_passed} test(s) failed. Check the setup.")
            return False
    
    def run_demo(self):
        """Run a quick demo of Grok 4"""
        logger.info("🚀 Running Grok 4 demo...")
        
        try:
            # Import and run the demo
            sys.path.append(str(self.root_dir))
            from grok4_flo_ai import main
            
            logger.info("Starting Grok 4 demo...")
            main()
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            return False
    
    def setup_all(self):
        """Run all setup steps"""
        logger.info("🔥 Running complete Grok 4 setup...")
        
        steps = [
            ("Installing dependencies", self.install_dependencies),
            ("Compiling lattice", self.compile_lattice),
            ("Setting up weights", self.download_weights),
            ("Configuring Flo AI", self.setup_flo_ai),
            ("Creating configs", self.create_config),
            ("Running tests", self.run_tests)
        ]
        
        success_count = 0
        for step_name, step_func in steps:
            logger.info(f"\n{'='*60}")
            logger.info(f"Step: {step_name}")
            logger.info('='*60)
            
            if step_func():
                success_count += 1
                logger.info(f"✅ {step_name} completed successfully")
            else:
                logger.error(f"❌ {step_name} failed")
        
        logger.info(f"\n🎯 Setup Summary: {success_count}/{len(steps)} steps completed")
        
        if success_count == len(steps):
            logger.info("🎉 Grok 4 setup completed successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Copy .env.example to .env and configure as needed")
            logger.info("2. Run: python grok4_flo_ai.py")
            logger.info("3. Check the documentation for advanced usage")
            return True
        else:
            logger.warning("⚠️ Setup completed with some errors. Check the logs above.")
            return False

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Grok 4 Setup Script")
    parser.add_argument("--install-deps", action="store_true", help="Install Python dependencies")
    parser.add_argument("--compile-lattice", action="store_true", help="Compile C transformer lattice")
    parser.add_argument("--download-weights", action="store_true", help="Setup Grok-1 base weights")
    parser.add_argument("--setup-flo-ai", action="store_true", help="Setup Flo AI framework")
    parser.add_argument("--create-config", action="store_true", help="Create configuration files")
    parser.add_argument("--run-tests", action="store_true", help="Run test suite")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--all", action="store_true", help="Run all setup steps")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup = Grok4Setup()
    
    if args.all:
        setup.setup_all()
    else:
        if args.install_deps:
            setup.install_dependencies()
        if args.compile_lattice:
            setup.compile_lattice()
        if args.download_weights:
            setup.download_weights()
        if args.setup_flo_ai:
            setup.setup_flo_ai()
        if args.create_config:
            setup.create_config()
        if args.run_tests:
            setup.run_tests()
        if args.demo:
            setup.run_demo()
        
        # If no specific args, show help
        if not any([args.install_deps, args.compile_lattice, args.download_weights, 
                   args.setup_flo_ai, args.create_config, args.run_tests, args.demo]):
            parser.print_help()
            print("\n🚀 Quick start: python setup_grok4.py --all")

if __name__ == "__main__":
    main()
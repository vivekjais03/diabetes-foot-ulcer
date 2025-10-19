#!/usr/bin/env python3
"""
Quick Start Script for Foot Ulcer Detection Application
This script will help you get everything running quickly!
"""

import os
import sys
import subprocess
import time

def print_banner():
    """Print application banner"""
    print("=" * 70)
    print("🦶 FOOT ULCER DETECTION & ANALYSIS SYSTEM")
    print("=" * 70)
    print("🚀 Quick Start Script - Getting You Up and Running!")
    print("=" * 70)

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        print("💡 Try running manually: pip install -r requirements.txt")
        return False

def check_model():
    """Check if trained model exists"""
    print("\n🔍 Checking for trained model...")
    
    model_path = "models/foot_ulcer_model.h5"
    
    if os.path.exists(model_path):
        print("✅ Trained model found!")
        return True
    else:
        print("❌ No trained model found!")
        print("💡 You need to train the model first.")
        return False

def train_model():
    """Offer to train the model"""
    print("\n🎯 Model Training Required")
    print("You need to train the model before running the application.")
    
    response = input("Would you like to train the model now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\n🚀 Starting model training...")
        print("⚠️  This may take several minutes depending on your hardware.")
        
        try:
            # Check if training script exists
            if os.path.exists("notebooks/train_model.py"):
                print("✅ Training script found, starting training...")
                
                # Run training
                result = subprocess.run([sys.executable, "notebooks/train_model.py"], 
                                     capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ Model training completed successfully!")
                    return True
                else:
                    print("❌ Model training failed!")
                    print("Error output:", result.stderr)
                    return False
            else:
                print("❌ Training script not found!")
                return False
                
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    else:
        print("💡 You can train the model later using: python notebooks/train_model.py")
        return False

def run_tests():
    """Run system tests"""
    print("\n🧪 Running system tests...")
    
    try:
        if os.path.exists("test_app.py"):
            result = subprocess.run([sys.executable, "test_app.py"], 
                                 capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ All tests passed!")
                return True
            else:
                print("❌ Some tests failed!")
                print("Test output:", result.stdout)
                return False
        else:
            print("⚠️  Test script not found, skipping tests")
            return True
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def start_application():
    """Start the Flask application"""
    print("\n🚀 Starting Foot Ulcer Detection Application...")
    print("=" * 50)
    print("📱 The application will open in your web browser")
    print("🌐 URL: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        # Start the Flask app
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to start application: {e}")

def main():
    """Main quick start function"""
    print_banner()
    
    # Step 1: Check Python version
    if not check_python_version():
        return False
    
    # Step 2: Install requirements
    if not install_requirements():
        return False
    
    # Step 3: Check for model
    if not check_model():
        if not train_model():
            print("\n❌ Cannot proceed without a trained model!")
            print("💡 Please train the model manually and try again.")
            return False
    
    # Step 4: Run tests
    if not run_tests():
        print("\n⚠️  Tests failed, but continuing...")
    
    # Step 5: Start application
    start_application()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ Quick start failed!")
            print("💡 Please check the errors above and try again.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Quick start interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

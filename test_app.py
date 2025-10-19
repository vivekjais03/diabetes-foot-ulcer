#!/usr/bin/env python3
"""
Test script for Foot Ulcer Detection Application
Run this to verify all components are working correctly
"""

import os
import sys
import importlib.util

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    required_packages = [
        'tensorflow',
        'cv2',
        'numpy',
        'PIL',
        'matplotlib',
        'flask',
        'reportlab'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages: pip install -r requirements.txt")
        return False
    else:
        print("✅ All packages imported successfully!")
        return True

def test_model():
    """Test if the trained model exists and can be loaded"""
    print("\n🔍 Testing model availability...")
    
    model_path = "models/foot_ulcer_model.h5"
    
    if not os.path.exists(model_path):
        print(f"  ❌ Model not found: {model_path}")
        print("  Please train the model first using: python notebooks/train_model.py")
        return False
    
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        print(f"  ✅ Model loaded successfully: {model_path}")
        print(f"  📊 Model summary:")
        model.summary()
        return True
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        return False

def test_enhanced_model():
    """Test the enhanced model functionality"""
    print("\n🔍 Testing enhanced model...")
    
    try:
        from notebooks.enhanced_model import EnhancedFootUlcerModel
        
        model_path = "models/foot_ulcer_model.h5"
        if os.path.exists(model_path):
            enhanced_model = EnhancedFootUlcerModel(model_path)
            print("  ✅ Enhanced model initialized successfully!")
            return True
        else:
            print("  ❌ Base model not found, skipping enhanced model test")
            return False
    except Exception as e:
        print(f"  ❌ Enhanced model test failed: {e}")
        return False

def test_flask_app():
    """Test if Flask app can be imported and configured"""
    print("\n🔍 Testing Flask application...")
    
    try:
        # Temporarily modify sys.path to import app
        sys.path.insert(0, os.getcwd())
        
        # Test basic Flask functionality
        from flask import Flask
        app = Flask(__name__)
        print("  ✅ Flask application created successfully!")
        
        # Test if our app.py can be imported
        try:
            import app
            print("  ✅ Main application module imported successfully!")
            return True
        except Exception as e:
            print(f"  ❌ Main application import failed: {e}")
            return False
            
    except Exception as e:
        print(f"  ❌ Flask test failed: {e}")
        return False

def test_templates():
    """Test if HTML templates exist"""
    print("\n🔍 Testing HTML templates...")
    
    template_path = "templates/index.html"
    
    if not os.path.exists(template_path):
        print(f"  ❌ Template not found: {template_path}")
        return False
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'Foot Ulcer Detection' in content:
                print("  ✅ HTML template found and contains expected content!")
                return True
            else:
                print("  ❌ Template content validation failed")
                return False
    except Exception as e:
        print(f"  ❌ Template read failed: {e}")
        return False

def test_dataset():
    """Test if dataset structure is correct"""
    print("\n🔍 Testing dataset structure...")
    
    required_dirs = [
        "dataset/split_dataset/train/Normal(Healthy skin)",
        "dataset/split_dataset/train/Abnormal(Ulcer)",
        "dataset/split_dataset/val/Normal(Healthy skin)",
        "dataset/split_dataset/val/Abnormal(Ulcer)"
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            file_count = len([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  ✅ {dir_path}: {file_count} images")
        else:
            print(f"  ❌ {dir_path}: Not found")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"  ⚠️  Missing directories: {', '.join(missing_dirs)}")
        return False
    else:
        print("  ✅ Dataset structure is correct!")
        return True

def main():
    """Run all tests"""
    print("🚀 Foot Ulcer Detection Application - System Test")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Model Availability", test_model),
        ("Enhanced Model", test_enhanced_model),
        ("Flask Application", test_flask_app),
        ("HTML Templates", test_templates),
        ("Dataset Structure", test_dataset)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your application is ready to run.")
        print("\n🚀 To start the application:")
        print("   python app.py")
        print("   Then open: http://localhost:5000")
    else:
        print("⚠️  Some tests failed. Please fix the issues before running the application.")
        
        if not any(name == "Model Availability" and result for name, result in results):
            print("\n💡 To train the model:")
            print("   python notebooks/train_model.py")
        
        if not any(name == "Package Imports" and result for name, result in results):
            print("\n💡 To install dependencies:")
            print("   pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

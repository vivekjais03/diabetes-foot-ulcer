import requests
import json

def test_upload_endpoint():
    print("Testing upload endpoint with sample image...")
    try:
        files = {'file': open('dataset/split_dataset/test/Abnormal(Ulcer)/1.jpg', 'rb')}
        data = {'confidence_threshold': '0.8'}
        response = requests.post('http://localhost:5000/upload', files=files, data=data)
        print("Status: {}".format(response.status_code))

        if response.status_code == 200:
            result = response.json()
            print("Class: {}".format(result['class']))
            print("Confidence: {}".format(result.get('confidence', 'N/A')))
            print("Remedies count: {}".format(len(result['remedies']['remedies'])))
            print("XAI explanation available: {}".format(bool(result.get('xai_explanation'))))
            return True
        else:
            print("Error: {}".format(response.text))
            return False
    except Exception as e:
        print("Exception: {}".format(e))
        return False

def test_chatbot():
    print("\nTesting chatbot endpoint...")
    try:
        data = {'message': 'infection', 'language': 'en'}
        response = requests.post('http://localhost:5000/chatbot', json=data)
        print("Status: {}".format(response.status_code))

        if response.status_code == 200:
            result = response.json()
            print("Response length: {} characters".format(len(result['response'])))
            return True
        else:
            print("Error: {}".format(response.text))
            return False
    except Exception as e:
        print("Exception: {}".format(e))
        return False

if __name__ == "__main__":
    print("Running endpoint tests...\n")

    upload_success = test_upload_endpoint()
    chatbot_success = test_chatbot()

    print("\nTest Results:")
    print("Upload endpoint: {}".format('PASS' if upload_success else 'FAIL'))
    print("Chatbot endpoint: {}".format('PASS' if chatbot_success else 'FAIL'))

    if upload_success and chatbot_success:
        print("\n✅ All critical endpoints are working correctly!")
    else:
        print("\n❌ Some endpoints failed testing.")

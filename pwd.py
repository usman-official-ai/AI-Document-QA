# Mock pwd module for Windows 
import os 
def getpwuid(uid): 
    return type('obj', (object,), {'pw_name': 'user'})() 

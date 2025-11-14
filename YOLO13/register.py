import requests

import time
import os


def register_web():
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.18"
                      "3 Safari/537.36 Edg/86.0.622.63"
    }
    url = "http://192.168.255.247/1.htm"
    data = {
        "DDDDD": "50023420010729685X",
        "upass": "64a209c0f284e769a3920cedf9c77648123456781",
        "R1": "0",
        "R2": "1",
        "para": "00",
        "MKKey": "123456",
    }
    response = requests.post(url=url, headers=header, data=data)
    if response.status_code == 200:
        print("success!!!")


def isConnected():
    try:
        html = requests.get("http://47.108.59.218:8090", timeout=5)
        info = html.text
        print(info)

        if len(info) >20:
            return True
        else:
            print("noraml")
            return False
    except:
        print("break......")
        return True


if __name__ == '__main__':
    while True:
        if  isConnected():
            try:
                register_web()
            except:
                print("Network abnormality")
        time.sleep(10)

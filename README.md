# TM Research 1
#### Repository for theoretical mechanics' research 1

## Manual rendering

1. Make sure you have installed:
    - **Python 3.9+** (make sure command `python3 -V` or `python -V`)
    - **PIP** (make sure command `pip`)
    - **Python Venv** for setup virtual environment (check this [useful link](https://docs.python.org/3/library/venv.html))

2. Clone the repo into your folder:
    ```shell
    git clone https://github.com/someilay/TM_Research_1.git
    cd ./TM_Research_1
    ```

3. Setup virtual environment:
    ```shell
    python3 -m venv venv
    ```

4. Activate environment ([guide](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments))

5. Install [matplotlib](https://matplotlib.org), version 3.5.0
    ```shell
    pip install matplotlib==3.5.0
    ```

6. Render graphs by executing:
    ```shell
    python3 main.py
    ```

   Rendered graphs will be stored in the same directory. It can take a few minutes.
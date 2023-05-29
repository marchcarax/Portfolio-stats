import os
import fundamental_anal as fanal

def main():

    #Put the ticker to analyze:
    ticker = 'aapl'
    fanal.fundamental_data(ticker)

    #I like outputs...
    try:
        os.remove('"Fundamental analysis\\stock_fundamentals.txt"')
    except Exception:
        pass
    
    os.system('python ".\\Fundamental analysis\\fanal_main.py" > "Fundamental analysis\\stock_fundamentals.txt"')

if __name__ == '__main__':
    main()

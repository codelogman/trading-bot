import pandas as pd
import numpy as np
import talib as ta
import datetime
import json

import time
from scipy import stats
import sys
import colored
from colored import fore, back, style

from colored import fg, bg, attr

import chime

from datetime import datetime, timedelta
import simpleaudio as sa
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation as animation
from iqoptionapi.stable_api import IQ_Option
import matplotlib.ticker as ticker
import beepy
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import load_model
import pickle

#init(autoreset=True)

tolerance = 0.00001  # para que cualquier cosa no sea un nivel dejar en casi 0
tolerance_trend = 0.00001  #funciona en 00001 sin volumen
hibernate = False
time_to_hibernate = 3

trades = []

initial_investment = 30  # Monto de inversión inicial
investment_operation = 2  # Monto de operacion
initial_investment_constant = initial_investment
investment_operation_constant = investment_operation
investment_restructure = 0
pair = "EURUSD"

start_time = datetime.now()

## modelo de aprendizaje
#restructured
def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)

def load_trained_model(model_path):
    return load_model(model_path)

def predict_trade(model, sequence):
    prediction = model.predict(sequence)
    return 1 if prediction > 0.49 else 0

def save_candles_to_csv(df, start_date, end_date, operation_result, file_name="candle_data.csv"):
    candle_data = df.loc[start_date:end_date, ['open', 'max', 'min', 'close']]
    candle_data['result'] = operation_result
    candle_data.to_csv(file_name, mode="a", header=not os.path.exists(file_name))


def get_balance(api):
    return api.get_balance()               

def is_exhaustion_candle(current_candle, previous_candles, exhaustion_ratio=3):
    avg_body_size = sum(abs(candle['close'] - candle['open']) for candle in previous_candles) / len(previous_candles)
    current_body_size = abs(current_candle['close'] - current_candle['open'])
    if current_body_size > avg_body_size * exhaustion_ratio:
        return True
    return False

## menu de opciones:

def menu(capital, monto_operacion):

    time_to_hibernate = 10
    hibernate = False
    start_time = datetime.now()
    
    while True:
        print("\n Menú de opciones:")
        print(" 1. Modificar capital inicial")
        print(" 2. Modificar monto de operación")
        print(" 3. Poner el script a hibernar")
        print(" 4. Salir de Hibernacion")
        print(" 5. Continuar")

        option = input(" Seleccione una opción: ")

        if option == '1':
            capital = float(input(" Ingrese el nuevo capital inicial: "))
            print(f" Capital inicial actualizado: {capital}")
        elif option == '2':
            monto_operacion = float(input(" Ingrese el nuevo monto de operación: "))
            print(f" Monto de operación actualizado: {monto_operacion}")
        elif option == '3':
            time_to_hibernate = int(input(" Tiempo para hibernar en minutos: "))
            hibernate = True
            start_time = datetime.now()
            break
        elif option == '4':
            print(" Saliendo de hibernacion...\n\n")
            hibernate = False
            break
        elif option == '5':
            trade_done = False
            break
        else:
            print("Opción inválida, por favor intente de nuevo.")

    return capital, monto_operacion, hibernate, time_to_hibernate, start_time, trade_done

## progress bar

def print_progress_bar(current_second):
    if current_second < 59:
        progress = int(current_second / 59 * 31)
        sys.stdout.write("\r realtime: [{0}{1}] {2}/{3}  ".format('▯' * progress, ' ' * (31 - progress), current_second, 59))
        sys.stdout.flush()
    else:
        sys.stdout.write("\r" + " " * 51)
        sys.stdout.flush()


def print_progress_barV2(current_second):
    if current_second < 59:
        #animation_chars = "←↖↑↗→↘↓↙"
        animation_chars = "⠋⠙⠚⠞⠖⠦⠴⠲⠳⠓"
        current_animation_char = animation_chars[current_second % len(animation_chars)]
        sys.stdout.write("\r start in: {0}  {1}|{2}       ".format(current_animation_char, current_second, 59))
        sys.stdout.flush()
    else:
        sys.stdout.write("\r" + " " * 25)
        sys.stdout.flush()

def print_progress_bar_hibernate(current_minute, total_minutes=30):
    if current_minute < total_minutes:
        progress = int(current_minute / total_minutes * 31)
        sys.stdout.write("\r hibernating: [{0}{1}] {2}/{3} minutos ".format('▮' * progress, ' ' * (31 - progress), current_minute, total_minutes))
        sys.stdout.flush()
    else:
        sys.stdout.write("\r" + " " * 59)
        sys.stdout.flush()

#levels, try to test trader Kanvel diagonal levels

def find_levels(df, min_touches_2, min_touches_3, tolerance, price_color, current_candle):
    levels_2_touches = []
    levels_3_touches = []
    broken_levels = []
    n = len(df)

    for index in range(n - 1, 1, -1):
        support_level = df.iloc[index]['min']
        resistance_level = df.iloc[index]['max']

        support_touches = 0
        resistance_touches = 0

        for i in range(index - 1, -1, -1):
            # Si un soporte roto se convierte en resistencia
            if ('support', support_level) in broken_levels and support_level - tolerance <= df.iloc[i]['max'] <= support_level + tolerance:
                resistance_touches += 1

            # Si una resistencia rota se convierte en soporte
            if ('resistance', resistance_level) in broken_levels and resistance_level - tolerance <= df.iloc[i]['min'] <= resistance_level + tolerance:
                support_touches += 1

            # Verificar el color de las velas y si cumplen con las condiciones de soporte y resistencia
            if support_level - tolerance <= df.iloc[i]['min'] <= support_level + tolerance and price_color[i] == 'red' and price_color[i - 1] == 'green':
                support_touches += 1
            if resistance_level - tolerance <= df.iloc[i]['max'] <= resistance_level + tolerance and price_color[i] == 'green' and price_color[i - 1] == 'red':
                resistance_touches += 1

            if support_touches > min_touches_3 and resistance_touches > min_touches_3:
                break

        #current_price = df.iloc[-1]['close']
        current_price = current_candle['close']

        # Agregar niveles rotos a la lista de broken_levels
        if support_touches == min_touches_3 and df.iloc[index]['min'] < support_level - tolerance:
            broken_levels.append(('support', support_level))
        if resistance_touches == min_touches_3 and df.iloc[index]['max'] > resistance_level + tolerance:
            broken_levels.append(('resistance', resistance_level))

        # Agregar soportes por debajo del precio actual y resistencias por encima del precio actual
        if support_touches == min_touches_2 and support_level < current_price and ('support', support_level) not in levels_2_touches:
            levels_2_touches.append(('support', support_level))
        if resistance_touches == min_touches_2 and resistance_level > current_price and ('resistance', resistance_level) not in levels_2_touches:
            levels_2_touches.append(('resistance', resistance_level))

        if support_touches == min_touches_3 and support_level < current_price and ('support', support_level) not in levels_3_touches:
            levels_3_touches.append(('support', support_level))
        if resistance_touches == min_touches_3 and resistance_level > current_price and ('resistance', resistance_level) not in levels_3_touches:
            levels_3_touches.append(('resistance', resistance_level))

    return levels_2_touches, levels_3_touches

def is_breaking_level(df, current_candle, previous_candle, levels, tolerance):
    breaking_resistance = None
    breaking_support = None

    for level_type, level_value in levels:
        if level_type == "resistance":
            if previous_candle['max'] <= level_value and current_candle['max'] > level_value + tolerance:
                breaking_resistance = level_value
        elif level_type == "support":
            if previous_candle['min'] >= level_value and current_candle['min'] < level_value - tolerance:
                breaking_support = level_value

    return breaking_resistance, breaking_support

def is_touching_level(df, levels, tolerance, price_color, current_candle_t, min_touches=2, close_tolerance=0.000003):
    previous_candle = df.iloc[-2]
    #current_candle = df.iloc[-1]
    current_candle = current_candle_t
    resistance_touch = False
    support_touch = False

    for level_type, level_value in levels:
        touches = 0
        for index in range(len(df) - 2, -1, -1):  # Comienza en la vela anterior a la actual
            if level_type == "resistance":
                if df.iloc[index]['max'] >= level_value - tolerance and df.iloc[index]['max'] <= level_value + tolerance:
                    touches += 1
                if touches == min_touches:
                    if previous_candle['max'] >= level_value - tolerance and previous_candle['max'] <= level_value + tolerance:
                        if current_candle['max'] <= level_value + tolerance and price_color[-2] == 'green':
                            if previous_candle['close'] >= level_value - close_tolerance and previous_candle['close'] <= level_value + close_tolerance:
                                resistance_touch = True
                                break
            elif level_type == "support":
                if df.iloc[index]['min'] >= level_value - tolerance and df.iloc[index]['min'] <= level_value + tolerance:
                    touches += 1
                if touches == min_touches:
                    if previous_candle['min'] >= level_value - tolerance and previous_candle['min'] <= level_value + tolerance:
                        if current_candle['min'] >= level_value - tolerance and price_color[-2] == 'red':
                            if previous_candle['close'] >= level_value - close_tolerance and previous_candle['close'] <= level_value + close_tolerance:
                                support_touch = True
                                break

    return resistance_touch, support_touch


with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=RuntimeWarning)
    # La parte de tu código donde deseas desactivar los warnings


def is_trendline(candles, trend_type, min_touches=3, tolerance_trend=0.0001):
    if len(candles) < min_touches:
        return False

    start_time = candles.iloc[0]['at']

    if trend_type == "ascending":
        for _, c in candles.iterrows():
            if c['close'] <= c['open']:
                return False
        corners = [(int(((c['at'] - start_time).total_seconds()) // (timeframe * 60)), c['min']) for _, c in candles.iterrows()]

    elif trend_type == "descending":
        for _, c in candles.iterrows():
            if c['close'] >= c['open']:
                return False
        corners = [(int(((c['at'] - start_time).total_seconds()) // (timeframe * 60)), c['max']) for _, c in candles.iterrows()]

    try:
        m, b = np.polyfit([p[0] for p in corners], [p[1] for p in corners], 1)
    except:
        print(fore.YELLOW_2 + back.RED + style.BOLD + " Excepcion:" + style.RESET + fore.YELLOW_1 + " polyfit encontro muy poca variación _|" + style.RESET)
        return False

    if (trend_type == "ascending" and m <= 0) or (trend_type == "descending" and m >= 0):
        return False

    for idx, (_, c) in enumerate(candles.iterrows()):
        price = c['min'] if trend_type == "ascending" else c['max']
        if abs(price - (m * (int(((c['at'] - start_time).total_seconds()) // (timeframe * 60))) + b)) > tolerance_trend:
            return False

    return True

def is_trendline_current(candles, trend_type, min_touches=3, tolerance_trend=0.0001):
    if len(candles) < min_touches:
        return False, None

    start_time = candles.iloc[0]['at']

    if trend_type == "ascending":
        for idx, c in candles.iterrows():
            if c['close'] <= c['open']:
                return False, None
            #min_price = c['min'] if abs(c['max'] - c['min']) >= tolerance_trend else c['open']
            corners = [(int(((c['at'] - start_time).total_seconds()) // (timeframe * 60)), c['min']) for _, c in candles.iterrows()]

    elif trend_type == "descending":
        for idx, c in candles.iterrows():
            if c['close'] >= c['open']:
                return False, None
            #max_price = c['max'] if abs(c['max'] - c['min']) >= tolerance_trend else c['open']
            corners = [(int(((c['at'] - start_time).total_seconds()) // (timeframe * 60)), c['max']) for _, c in candles.iterrows()]
    #print(corners)
    try:
        m, b = np.polyfit([p[0] for p in corners], [p[1] for p in corners], 1)
        #print("valor de m: " + str(m) + "  valor de b: " + str(b))
    except:
        print(fore.YELLOW_2 + back.RED + style.BOLD + " Excepcion:" + style.RESET + fore.YELLOW_1 + " polyfit encontro muy poca variación _/" + style.RESET)
        # Comprobar si las dos últimas velas completas son verdes (precio de cierre > precio de apertura)
        if candles.iloc[-2]['close'] > candles.iloc[-2]['open'] and candles.iloc[-3]['close'] > candles.iloc[-3]['open']:
            # Si ambas velas son verdes, retornar False y 'put'
            return False, 'put'
        # Comprobar si las dos últimas velas completas son rojas (precio de cierre < precio de apertura)
        elif candles.iloc[-2]['close'] < candles.iloc[-2]['open'] and candles.iloc[-3]['close'] < candles.iloc[-3]['open']:
            # Si ambas velas son rojas, retornar False y 'call'
            return False, 'call'
        else:
            # Si las dos últimas velas completas no son ni todas verdes ni todas rojas, retornar False y None
            return False, None
    if (trend_type == "ascending" and m <= 0) or (trend_type == "descending" and m >= 0):
        return False, None

    for idx, (_, c) in enumerate(candles.iterrows()):
        price = c['min'] if trend_type == "ascending" else c['max']
        if abs(price - (m * (int(((c['at'] - start_time).total_seconds()) // (timeframe * 60))) + b)) > tolerance_trend:
            return False, None

    return True, None


def find_trendlines(df, trend_type="ascending", min_touches=3, tolerance=tolerance_trend):
    trendlines = []
    for start_index in range(len(df) - min_touches):
        for end_index in range(start_index + min_touches - 1, len(df)):
            candles = df.iloc[start_index:end_index + 1]

            if is_trendline(candles, trend_type, min_touches, tolerance):
                if trend_type == "ascending":
                    corners = [(int(((c['at'] - candles.iloc[0]['at']).total_seconds()) // (timeframe * 60)), c['min'] if c['close'] > c['open'] else c['close']) for _, c in candles.iterrows()]
                elif trend_type == "descending":
                    corners = [(int(((c['at'] - candles.iloc[0]['at']).total_seconds()) // (timeframe * 60)), c['max'] if c['close'] < c['open'] else c['close']) for _, c in candles.iterrows()]

                m, b = np.polyfit([p[0] for p in corners], [p[1] for p in corners], 1)

                trendlines.append((trend_type, m, b, start_index, end_index))

    return trendlines

def is_touching_trendline(df, trend_type, current_candle, tolerance_trend=tolerance_trend):
    trendlines = find_trendlines(df, trend_type, min_touches=3, tolerance=tolerance_trend)

    for trendline in trendlines:
        _, m, b, start_index, end_index = trendline
        start_time = df.iloc[start_index]['at']        

        if trend_type == "ascending":
            price = current_candle['min']
        elif trend_type == "descending":
            price = current_candle['max']

        # Comprobar si el precio de la vela actual está dentro de la tolerancia de la línea de tendencia
        index_difference = round(((current_candle['at'] - start_time).total_seconds()) / (timeframe * 60))
        estimated_price = m * index_difference + b
        price_difference = abs(price - estimated_price)

        if price_difference <= tolerance_trend:
            return True

    return False

def is_last_trendline_self(df, trend_type="ascending", min_touches=3, tolerance=tolerance_trend):
    last_candles = df.iloc[-min_touches:].copy()  # Obtiene las últimas (min_touches) velas del DataFrame incluyendo la actual    
    return is_trendline_current(last_candles, trend_type, min_touches, tolerance)


def is_last_trendline(df, trend_type="ascending", min_touches=3, tolerance=tolerance_trend):
    last_candles = df.iloc[-min_touches: -1].copy()  # Obtiene las últimas (min_touches - 1) velas del DataFrame
    return is_trendline_current(last_candles, trend_type, 3, tolerance)

## nueva implementacion:

def record_trade(trades_list):
    wins = trades_list.count("win")
    losses = trades_list.count("loss")
    total_trades = len(trades_list)
    
    win_percentage = (wins / total_trades) * 100
    loss_percentage = (losses / total_trades) * 100

    return win_percentage, loss_percentage, total_trades


def get_current_candle(api, pair, timeframe):
    candles = api.get_candles(pair, (timeframe * 60), 1, time.time())
    current_time = time.time()
    candle_index = int(current_time // (timeframe * 60))
    candle_at = datetime.fromtimestamp(current_time)
    
    current_candle = {
        'index': candle_index,
        'at': candle_at,
        'min': candles[0]['min'],
        'max': candles[0]['max'],
        'open': candles[0]['open'],
        'close': candles[0]['close'],
    }
    return current_candle


def get_previous_candle(api, pair, timeframe):
    candles = api.get_candles(pair, (timeframe * 60), 2, time.time())
    current_time = time.time()
    candle_index = int(current_time // (timeframe * 60))
    candle_at = datetime.fromtimestamp(current_time)
    
    previous_candle = {
        'index': candle_index,
        'at': candle_at,    
        'min': candles[1]['min'],
        'max': candles[1]['max'],
        'open': candles[1]['open'],
        'close': candles[1]['close'],
    }
    return previous_candle


def candle_color(candle):
    if candle['open'] < candle['close']:
        return 'g'
    elif candle['open'] > candle['close']:
        return 'r'
    else:
        return 'd'

def plot_colored_candles(df, ax, trendlines):
    for num_index, (open_price, high, low, close_price) in enumerate(zip(df['open'], df['max'], df['min'], df['close'])):
        current_candle = pd.Series({'open': open_price, 'max': high, 'min': low, 'close': close_price})

        e_facecolor = None
        edgecolor = None
        is_within_trendline = False
        for trend_type, m, b, start_index, end_index in trendlines:
            if start_index - 1 <= num_index <= end_index - 1:
                is_within_trendline = True
                break

        color = "green" if close_price >= open_price else "red"

        e_facecolor = "green" if close_price >= open_price else "red"
        
        if is_within_trendline and close_price >= open_price:
            e_facecolor = "darkgreen" 
        if is_within_trendline and close_price <= open_price:
            e_facecolor = "orangered" 
                
        value_candle = open_price - close_price
        
        if value_candle == 0:
            value_candle = 0.000004
            e_facecolor = "grey"
            
        
        rect = plt.Rectangle((num_index - 0.3, min(open_price, close_price)), 0.6, abs(value_candle), facecolor=e_facecolor, edgecolor=edgecolor)
        ax.add_patch(rect)

        ax.vlines(num_index, low, high, color=e_facecolor)        

# interes compuesto y reestructuracion de capital

def update_investment_and_check_stop_loss(valor, investment, initial_investment, target_amount, min_capital_allowed, restructure_ratio, compound_interest_active, consecutive_wins, hibernate, time_to_hibernate, start_time):

    global initial_investment_constant
    global investment_operation_constant
    global exchange_break
    global investment_restructure
    
    hibernate = False
    time_to_hibernate = 10
    start_time = datetime.now()

    win_percentage, loss_percentage, total_trades = record_trade(trades)        
    
    if valor > 0:  # Si ganó la operación
        if consecutive_wins == 1:  # Si tiene una victorias consecutivas
            if win_percentage > 60 and total_trades > 0:
                compound_interest_active = True
                print("\n interes compuesto " + fore.GREEN + "activo" + style.RESET)
        # ring! interes compuesto de 2 + 1
        if consecutive_wins != 3:
            try:
                w_object = sa.WaveObject.from_wave_file('money.wav')
                p_object = w_object.play()
                p_object.wait_done()
            except FileNotFoundError:
                print("Wav File does not exists")
    
    else:  # Si perdió la operación
        consecutive_wins = 0  # Resetear conteo de ganadas consecutivas
        if compound_interest_active:
            print("\n interes compuesto " + fore.ORANGE_RED_1 + "desactivado" + style.RESET)
        compound_interest_active = False

    total_amount = valor
    initial_investment += total_amount
    if initial_investment >= target_amount:
        initial_investment = initial_investment_constant
        investment_restructure += 1
        new_investment = investment_operation_constant

    if compound_interest_active and consecutive_wins >= 1 and consecutive_wins < 4 and win_percentage > 60 and total_trades > 0:
        new_investment = total_amount + investment        
        if consecutive_wins == 3 and compound_interest_active == True:
            print(" combo breaker " + fore.GREEN + "activo" + style.RESET)
    else:
        new_investment = investment_operation_constant
        if consecutive_wins == 3:
           # monster!
           try:
               w_object = sa.WaveObject.from_wave_file('nuke.wav')
               p_object = w_object.play()
               p_object.wait_done()
           except FileNotFoundError:
               print("Wav File does not exists")
               
        consecutive_wins = 0        


    print(fore.GREEN + " wins" + fore.WHITE + f": {round(win_percentage,2)} %    " + fore.RED + "loss" + fore.WHITE + f": {round(loss_percentage,2)} %" + f" total trades: {total_trades}" + style.RESET)
    if initial_investment <= min_capital_allowed:
        print("\nStop loss alcanzado. Deteniendo el script.")
        sys.exit()

    if consecutive_wins >= 3:
        hibernate = True
        start_time = datetime.now()
        time_to_hibernate = 11
        beepy.beep(sound=1)
        
    if consecutive_loses == 2:
        hibernate = True
        start_time = datetime.now()
        time_to_hibernate = 11
        beepy.beep(sound=1)
    
        
    if consecutive_loses > 2:
        print(fore.YELLOW_2 + back.RED + "ALERTA" + style.RESET + fore.YELLOW_1 + " por rango o canal, se apagan las estrategias" + style.RESET)
        hibernate = True
        start_time = datetime.now()
        time_to_hibernate = 30
        try:
            # Define object to play
            w_object = sa.WaveObject.from_wave_file('hadouken.wav')
            # Define object to control the play
            p_object = w_object.play()
            p_object.wait_done()
        # Print error message if the file does not exist
        except FileNotFoundError:
            print("Wav File does not exists")
        
    return round(initial_investment,2), round(target_amount,2), round(new_investment,2), consecutive_wins, compound_interest_active, hibernate, time_to_hibernate, start_time


def toggle_chart_visibility(fig, is_chart_visible):
    fig.set_visible(is_chart_visible)
    fig.canvas.draw_idle()
    return not is_chart_visible


def time_formatter(x, pos=None):
    if x < 0 or x >= len(df.index):
        return ''
    at_value = df['at'].iloc[int(x)]
    at_datetime = pd.to_datetime(at_value)
    return at_datetime.strftime('%M')

           
# Configura tus credenciales de IQOption aquí

print(fore.GREEN + '''
      +-++-++-++-++-+ +-+  +-++-++-++-++-++-+
      |p||r||o||t||o| |5|  |s||c||r||i||p||t|
      +-++-++-++-++-+ +-+  +-++-++-++-++-++-+     
''')
print(fore.LIGHT_GOLDENROD_2A + '''
                ACCION DEl PRECIO
                      por
                  alex_strange
 ---------------------------------------------
''')
print(fore.WHITE)

try:
    # Define object to play
    w_object = sa.WaveObject.from_wave_file('money.wav')
    # Define object to control the play
    p_object = w_object.play()
    print(" --= Cuanto dinero vas a ganar hoy?  =--\n")
    p_object.wait_done()

# Print error message if the file does not exist
except FileNotFoundError:
    print("Wav File does not exists")


api = IQ_Option("codelogman@gmail.com", "vfwgm3gryg")
api.connect()

# Verifica si la conexión fue exitosa
if api.check_connect():
    print("\n Conectado al broker exitosamente !\n")
    beepy.beep(sound=5)
else:
    print("Error al conectarse")
    api.connect()

# get all the active assets
active_assets = api.get_all_open_time()

# get binary options only
binary_options = active_assets['binary']

# collect the names of all open binary options
open_binary_options = [asset for asset in binary_options if binary_options[asset]['open']]

# print them out, separated by commas
print(" {" + fore.LIGHT_GOLDENROD_2A + ', '.join(open_binary_options) + fore.WHITE + "}")
        
pair = input('\n Activo para operar: ').upper()

# Establecer tipo de cuenta
tipo_cuenta = input('\n Que tipo de cuenta desea usar (REAL/PRACTICE)? : ').upper()  # PRACTICE / REAL /TOURNAMENT
api.change_balance(tipo_cuenta)

print(' Balance total en cuenta: ', get_balance(api)) #valor del balance en el tipo de cuenta

initial_investment = float(input(' Monto de Inversion: '))
initial_investment_constant = initial_investment

investment_operation = float(input(' Indique un valor para operar: '))
investment_operation_constant = investment_operation

timeframe = 1  # en minutos

restructure_ratio = 0.6  # Porcentaje de reestructuración (60%)
target_amount = initial_investment * (1 + restructure_ratio)  # Monto objetivo de reestructuración

stop_loss_percent = 0.45  # Porcentaje de stop loss (45%)
min_capital_allowed = initial_investment * (stop_loss_percent)  # Monto mínimo de capital permitido
compound_interest_active = False
consecutive_wins = 0
consecutive_loses = 0    
ex_consecutive_loses = 0
print(" iniciando sesion de trading en el broker...")

fig, ax = plt.subplots(figsize=(12, 5))

mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
s = mpf.make_mpf_style(marketcolors=mc, gridstyle=":", y_on_right=True, facecolor='#F0FFFF')

# Aplicar el color de fondo al eje
ax.set_facecolor(s['facecolor'])

plt.tight_layout()

plt.show(block=False)

formatted_tolerance_levels = format(tolerance, '.6f')
st_formated_level = str(formatted_tolerance_levels)
#st_formated_level =st_formated_level[0:6] + "[" + st_formated_level[6:8] + "]"

valor = 0
previous_balance = 0
balance = get_balance(api)
cons_wins = 0

fig.canvas.manager.set_window_title("(" + pair + ")" + " : zonas, max|min y trendLines  pip tolerance: " + st_formated_level + " balance: " + str(balance) + " cons_wins: " + str(cons_wins))

previous_candle_timestamp = None

is_chart_visible = True

status = False
valor = 0
id = 0
trade_done = False

chime.theme('material')

levels = []
levels_touched = []    
trendlines = []
ascending_trendlines = []
descending_trendlines = []
trendline_candles = []
horizontal_wick_zones = []

formatted_tolerance_levels = format(tolerance, '.6f')
formatted_tolerance_trends = format(tolerance_trend, '.6f')
          
tolerance = float(formatted_tolerance_levels)
tolerance_trend  = float(formatted_tolerance_trends)   

wick_time = 15

candles_data = api.get_candles(pair, timeframe * 60, 60, time.time())
df = pd.DataFrame(candles_data)
df['at'] = pd.to_datetime(df['at'])
df['time'] = df['at'].map(lambda x: datetime.fromtimestamp(x.timestamp()))
df.set_index('time', inplace=True)
price_color = ['green' if df.iloc[i]['close'] > df.iloc[i]['open'] else 'red' for i in range(len(df))]


# puntero al precio
price_line = None
price_text = None

current_candle = get_current_candle(api, pair, timeframe)

levels_touched, levels = find_levels(df, 2, 3, tolerance, price_color, current_candle)

ascending_trendlines = find_trendlines(df, trend_type="ascending", min_touches=3, tolerance=tolerance_trend)
descending_trendlines = find_trendlines(df, trend_type="descending", min_touches=3, tolerance=tolerance_trend)          
# Combina líneas de tendencia alcistas y bajistas       
trendlines = ascending_trendlines + descending_trendlines        

scaler_path = 'scaler.pkl'
scaler = load_scaler(scaler_path)

model_path = 'trained_model.h5'
model = load_trained_model(model_path)

hibernate = True
time_to_hibernate = 2
chime.success()

counter = 0

while True:
    try:
        current_time = datetime.now().time()
        seconds_remaining = 60 - current_time.second
        seconds_elapsed = current_time.second
                
        current_time_min = datetime.now()
        elapsed_time_min = current_time_min - start_time
        elapsed_minutes = int(elapsed_time_min.total_seconds() // 60)
        
            
        if seconds_elapsed <= 1: 
            entrar = True
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            
            if not hibernate:
                print("\n")
                print(f" {dt_string} - obteniendo datos ...")            

            candles_data = api.get_candles(pair, timeframe * 60, 60, time.time())
            df = pd.DataFrame(candles_data)
            df['at'] = pd.to_datetime(df['at'])
            df['time'] = df['at'].map(lambda x: datetime.fromtimestamp(x.timestamp()))
            df.set_index('time', inplace=True)
            price_color = ['green' if df.iloc[i]['close'] > df.iloc[i]['open'] else 'red' for i in range(len(df))]        
                       
        else:
            entrar = False 
            
        if trade_done and not hibernate:
            current_second = datetime.now().second
            print_progress_bar(current_second)
        elif not hibernate:
            current_second = datetime.now().second
            print_progress_barV2(current_second)            
        
        if hibernate:
             print_progress_bar_hibernate(elapsed_minutes, total_minutes=time_to_hibernate)    
            
        if elapsed_minutes >= time_to_hibernate and hibernate:
            hibernate = False
            
        if entrar and not hibernate:
            new_candle_data = api.get_candles(pair, timeframe * 60, 1, time.time())
            new_candle = pd.DataFrame(new_candle_data)
            new_candle['at'] = pd.to_datetime(new_candle['at'])
            new_candle['time'] = new_candle['at'].map(lambda x: datetime.fromtimestamp(x.timestamp()))
            new_candle.set_index('time', inplace=True)

            df = df.iloc[1:].append(new_candle)

            # Seleccionar solo las columnas necesarias para el gráfico de velas japonesas
            df_candlestick = df[['open', 'max', 'min', 'close']]
            df_candlestick.columns = ['Open', 'High', 'Low', 'Close']  # Cambiar el nombre de las columnas para mplfinance

            # carga los datos al modelo
            
            # Asumiendo que 'df' es tu DataFrame con las velas y que ya tienes un objeto MinMaxScaler 'scaler' entrenado
            num_candles_to_save = 20
            last_20_rows = df.tail(num_candles_to_save)
            normalized_data = scaler.transform(last_20_rows[['open', 'max', 'min', 'close']])
            current_sequence = np.array([normalized_data])
            
            ## genera la prediccion:
            trade_prediction = predict_trade(model, current_sequence)
            print(fore.WHITE + " optimizer=" + fore.GREEN + "Adam" + fore.WHITE + "(learning_rate=" + fore.RED + "0.002" + fore.WHITE + ") loss=" + fore.GREEN + "binary_crossentropy, " + fore.YELLOW + "prdct" + fore.WHITE + "=" + fore.WHITE + str(trade_prediction) + style.RESET)      
            
            # Crear df_mpf
            df_mpf = df_candlestick.copy()
            df_mpf['num_index'] = range(len(df_mpf))
        
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")       
            print(f" {dt_string} - analizando mercado ...")
        
            # Calcular las tendencias alcistas y bajistas
            current_candle_index = len(df) - 1
            previous_candle = df.iloc[-2]
            
            current_candle = get_current_candle(api, pair, timeframe)
            
            #levels_touched, levels = find_levels(df, 2, 3, tolerance, price_color, current_candle)
            #breaking_resistance, breaking_support = is_breaking_level(df, current_candle, previous_candle, levels, tolerance_trend)            
            #resistance_touch, support_touch = is_touching_level(df, levels_touched, tolerance, price_color=price_color, current_candle_t=current_candle, min_touches=2)
            
            breaking_resistance = None
            breaking_support = None
            resistance_touch = False
            support_touch = False
            
            #levels_touched, levels = find_levels(df, 2, 3, tolerance, price_color, current_candle)
            #levels = find_levels(df, min_touches=3, tolerance=tolerance, price_color=price_color)
            #levels_touched = find_levels(df, min_touches=2, tolerance=tolerance, price_color=price_color)            

            candle_last_color = candle_color(previous_candle)
            
            if candle_last_color == "g":
                print(" previous candle color:  " + fore.GREEN + "▒" + style.RESET)
           
            if candle_last_color == "r":
                print(" previous candle color:  " + fore.RED + "▒" + style.RESET)

            if candle_last_color == "d":
                print(" previous candle color:  " + fore.WHITE + "▒" + style.RESET)
                   
            is_touching_ascending_trendline, ex_touch_ascending = is_last_trendline_self(df, "ascending")
            is_touching_descending_trendline, ex_touch_descending = is_last_trendline_self(df, "descending")
            
            is_trendline_ascending, ex_trend_ascending = is_last_trendline(df, "ascending")
            is_trendline_descending, ex_trend_descending  = is_last_trendline(df, "descending")

            # Lista de excepciones en orden de prioridad
            exceptions = [ex_touch_ascending, ex_touch_descending, ex_trend_ascending, ex_trend_descending]
            print(exceptions)
            # Use next() para obtener el primer elemento no None de la lista (si existe), de lo contrario, será None
            exception = next((ex for ex in exceptions if ex is not None), None)
            
            e_previous_candles = df.iloc[-6:-2].to_records()

            # Verifica si la vela anterior es una vela de cansancio
            exhaustion_ratio = 4
            is_exhausted = is_exhaustion_candle(previous_candle, e_previous_candles, exhaustion_ratio)

            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")       
            print(f" {dt_string} - analisis finalizado ...")
        
            if trade_done:   
                    valor= get_balance(api) - previous_balance

                    if valor > 0:                       
                       consecutive_wins += 1
                       cons_wins = consecutive_wins
                       consecutive_loses = 0
                       ex_consecutive_loses = 0
                       trades.append("win")
                       trade_done = False
                    elif valor < 0:
                       consecutive_wins = 0
                       cons_wins = consecutive_wins
                       consecutive_loses += 1
                       ex_consecutive_loses += 1
                       trades.append("loss")
                       trade_done = False
                    elif valor == 0:
                       consecutive_wins = 0
                       cons_wins = consecutive_wins
                       consecutive_loses += 0
                       ex_consecutive_loses += 0
                       trade_done = False
                           
                       beepy.beep(sound=3)     
                       num_candles_to_save = 20
                       start_date = df.index[-num_candles_to_save]
                       end_date = df.index[-1]
                       operation_result = "loss"
                       csv_filename = "candle_data.csv"
                       save_candles_to_csv(df, start_date, end_date, operation_result, file_name=csv_filename)
                       print(" Se guardaron los " + fore.CYAN + "datos" + fore.WHITE + " correctamente")
                      
            
                    initial_investment, target_amount, investment_operation, consecutive_wins, compound_interest_active, hibernate, time_to_hibernate, start_time = update_investment_and_check_stop_loss(valor, investment_operation, initial_investment, target_amount, min_capital_allowed, restructure_ratio, compound_interest_active, consecutive_wins, hibernate, time_to_hibernate, start_time)
                    balance = get_balance(api)                    
                    trade_result = "WIN" if valor > 0 else "LOSS"
                    trade_summary = f" {trade_result}: Profit {round(valor,2)}, Next entry {investment_operation}, Current Capital {initial_investment}, Target {target_amount}, Real Balance {balance}, restructures {investment_restructure}"
                    trade_color = fore.GREEN if valor > 0 else (fore.YELLOW if valor < 0 else fore.ORANGE_1)
                    print(trade_color + trade_summary + style.RESET)
      

            if trade_prediction == 0 and not is_exhausted:

                dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")       

                if breaking_resistance is not None and trade_done == False:                    
                    direction = "call"
                    #status,id = api.buy_digital_spot(pair, investment_operation, direction, timeframe)
                    status, id = api.buy(investment_operation, pair, direction, timeframe)
                    print(f" {dt_string} -" + fore.GREEN + " call " + fore.WHITE + "- rompiendo nivel de " + fore.GREEN + "resistencia   " + fore.GREEN + "▲" + style.RESET)
                    trade_done = True
                    previous_balance = balance
                    #beepy.beep(sound=1)
                
                    ascending_trendlines = find_trendlines(df, trend_type="ascending", min_touches=3, tolerance=tolerance_trend)
                    descending_trendlines = find_trendlines(df, trend_type="descending", min_touches=3, tolerance=tolerance_trend)          
                    # Combina líneas de tendencia alcistas y bajistas       
                    trendlines = ascending_trendlines + descending_trendlines       


                if breaking_support is not None and trade_done == False:                    
                    direction = "put"
                    #status,id = api.buy_digital_spot(pair, investment_operation, direction, timeframe)
                    status, id = api.buy(investment_operation, pair, direction, timeframe)
                    print(f" {dt_string} -" + fore.RED + " put " + fore.WHITE + "- rompiendo nivel de " + fore.RED + "soporte   " + fore.RED + "▼" + style.RESET)
                    trade_done = True
                    previous_balance = balance
                    #beepy.beep(sound=1)

                    ascending_trendlines = find_trendlines(df, trend_type="ascending", min_touches=3, tolerance=tolerance_trend)
                    descending_trendlines = find_trendlines(df, trend_type="descending", min_touches=3, tolerance=tolerance_trend)          
                    # Combina líneas de tendencia alcistas y bajistas       
                    trendlines = ascending_trendlines + descending_trendlines                        
            

                if resistance_touch and trade_done == False:                    
                    direction = "put"
                    #status,id = api.buy_digital_spot(pair, investment_operation, direction, timeframe)
                    status, id = api.buy(investment_operation, pair, direction, timeframe)
                    print(f" {dt_string} -" + fore.RED + " put " + fore.WHITE + "- tercer toque nivel de " + fore.GREEN + "resistencia   " + fore.RED + "▼" + style.RESET)
                    trade_done = True
                    previous_balance = balance
                    #beepy.beep(sound=1)

                    ascending_trendlines = find_trendlines(df, trend_type="ascending", min_touches=3, tolerance=tolerance_trend)
                    descending_trendlines = find_trendlines(df, trend_type="descending", min_touches=3, tolerance=tolerance_trend)          
                    # Combina líneas de tendencia alcistas y bajistas       
                    trendlines = ascending_trendlines + descending_trendlines        

                if support_touch and trade_done == False:                    
                    direction = "call"
                    #status,id = api.buy_digital_spot(pair, investment_operation, direction, timeframe)
                    status, id = api.buy(investment_operation, pair, direction, timeframe)
                    print(f" {dt_string} -" + fore.GREEN + " call " + fore.WHITE + "- tercer toque nivel de " + fore.RED + "soporte   " + fore.GREEN + "▲" + style.RESET)
                    trade_done = True
                    previous_balance = balance
                    #beepy.beep(sound=1)
                
                    ascending_trendlines = find_trendlines(df, trend_type="ascending", min_touches=3, tolerance=tolerance_trend)
                    descending_trendlines = find_trendlines(df, trend_type="descending", min_touches=3, tolerance=tolerance_trend)          
                    # Combina líneas de tendencia alcistas y bajistas       
                    trendlines = ascending_trendlines + descending_trendlines       

                if is_touching_ascending_trendline and trade_done == False:                    
                    direction = "call"
                    #status,id = api.buy_digital_spot(pair, investment_operation, direction, timeframe)
                    status, id = api.buy(investment_operation, pair, direction, timeframe)
                    print(f" {dt_string} -" + fore.GREEN + " call " + fore.WHITE + "- microtendencia alcista   " + fore.GREEN + "▲" + style.RESET)
                    trade_done = True
                    previous_balance = balance
                    #beepy.beep(sound=1)

                    ascending_trendlines = find_trendlines(df, trend_type="ascending", min_touches=3, tolerance=tolerance_trend)
                    descending_trendlines = find_trendlines(df, trend_type="descending", min_touches=3, tolerance=tolerance_trend)          
                    # Combina líneas de tendencia alcistas y bajistas       
                    trendlines = ascending_trendlines + descending_trendlines        

                if is_trendline_ascending and trade_done == False:                    
                    direction = "call"
                    #status,id = api.buy_digital_spot(pair, investment_operation, direction, timeframe)
                    status, id = api.buy(investment_operation, pair, direction, timeframe)
                    print(f" {dt_string} -" + fore.GREEN + " call " + fore.WHITE + "- microtendencia alcista __ " + fore.GREEN + "▲" + style.RESET)
                    trade_done = True
                    previous_balance = balance
                    #beepy.beep(sound=1)

                    ascending_trendlines = find_trendlines(df, trend_type="ascending", min_touches=3, tolerance=tolerance_trend)
                    descending_trendlines = find_trendlines(df, trend_type="descending", min_touches=3, tolerance=tolerance_trend)          
                    # Combina líneas de tendencia alcistas y bajistas       
                    trendlines = ascending_trendlines + descending_trendlines        
                

                if is_touching_descending_trendline and trade_done == False:                    
                    direction = "put"                    
                    #status,id = api.buy_digital_spot(pair, investment_operation, direction, timeframe)
                    status, id = api.buy(investment_operation, pair, direction, timeframe)
                    print(f" {dt_string} -" + fore.RED + " put " + fore.WHITE + "- microtendencia bajista   " + fore.RED + "▼" + style.RESET)
                    trade_done = True
                    previous_balance = balance
                    #beepy.beep(sound=1)

                    ascending_trendlines = find_trendlines(df, trend_type="ascending", min_touches=3, tolerance=tolerance_trend)
                    descending_trendlines = find_trendlines(df, trend_type="descending", min_touches=3, tolerance=tolerance_trend)          
                    # Combina líneas de tendencia alcistas y bajistas       
                    trendlines = ascending_trendlines + descending_trendlines        

                if is_trendline_descending and trade_done == False:                    
                    direction = "put"
                    #status,id = api.buy_digital_spot(pair, investment_operation, direction, timeframe)
                    status, id = api.buy(investment_operation, pair, direction, timeframe)
                    print(f" {dt_string} -" + fore.RED + " put " + fore.WHITE + "- microtendencia bajista __ " + fore.RED + "▼" + style.RESET)
                    trade_done = True
                    previous_balance = balance
                    #beepy.beep(sound=1)

                    ascending_trendlines = find_trendlines(df, trend_type="ascending", min_touches=3, tolerance=tolerance_trend)
                    descending_trendlines = find_trendlines(df, trend_type="descending", min_touches=3, tolerance=tolerance_trend)          
                    # Combina líneas de tendencia alcistas y bajistas       
                    trendlines = ascending_trendlines + descending_trendlines
                    
                #if exception is not None and trade_done == False:
                    #direction = exception                
                    #status, id = api.buy(investment_operation, pair, direction, timeframe)
                    #if direction == "put":
                    #    print(f" {dt_string} -" + fore.YELLOW_2 + back.RED + style.BOLD + "put" + style.RESET + fore.YELLOW_1 + "   - excepcion encontrada " + fore.RED + "▼" + style.RESET)
                    #else:
                    #    print(f" {dt_string} -" + fore.YELLOW_2 + back.RED + style.BOLD + "call" + style.RESET + fore.YELLOW_1 + "   - excepcion encontrada " + fore.GREEN + "▲" + style.RESET)
                    #trade_done = True
                    #previous_balance = balance
                    #beepy.beep(sound=1)

                    #ascending_trendlines = find_trendlines(df, trend_type="ascending", min_touches=3, tolerance=tolerance_trend)
                    #descending_trendlines = find_trendlines(df, trend_type="descending", min_touches=3, tolerance=tolerance_trend)          
                    # Combina líneas de tendencia alcistas y bajistas       
                    #trendlines = ascending_trendlines + descending_trendlines
                
            else:
                if is_exhausted and trade_done == False:
                    print(" vela de fuerza detectada !, se" + fore.RED + " anulan " + fore.WHITE + "operaciones ..." + style.RESET)
                    beepy.beep(sound=3)
                if trade_prediction != 0:
                    print(fore.LIGHT_BLUE + back.RED + style.BOLD + " Excepcion encontrada !" + style.RESET + fore.WHITE + " se anulan operaciones ..." + style.RESET)        
                    beepy.beep(sound=3)
                    
    ### hasta aqui llega la condicion if entrar ...    

        current_second = datetime.now().second
        if current_second > 20 and current_second <= 22:
        
            candles_data = api.get_candles(pair, timeframe * 60, 60, time.time())
            df = pd.DataFrame(candles_data)
            df['at'] = pd.to_datetime(df['at'])
            df['time'] = df['at'].map(lambda x: datetime.fromtimestamp(x.timestamp()))
            df.set_index('time', inplace=True)
            price_color = ['green' if df.iloc[i]['close'] > df.iloc[i]['open'] else 'red' for i in range(len(df))]
        
            ascending_trendlines = find_trendlines(df, trend_type="ascending", min_touches=3, tolerance=tolerance_trend)
            descending_trendlines = find_trendlines(df, trend_type="descending", min_touches=3, tolerance=tolerance_trend)          
            # Combina líneas de tendencia alcistas y bajistas       
            trendlines = ascending_trendlines + descending_trendlines
            balance = get_balance(api)
            fig.canvas.manager.set_window_title("(" + pair + ")" + " : zonas, max|min y trendLines  pip tolerance: " + st_formated_level + " balance: " + str(balance) + " cons_wins: " + str(cons_wins))
            
        
        ### leemos la ultima vela para mantener el grafico en linea ...
        new_candle_data = api.get_candles(pair, timeframe * 60, 1, time.time())
        new_candle = pd.DataFrame(new_candle_data)
        new_candle['at'] = pd.to_datetime(new_candle['at'])
        new_candle['time'] = new_candle['at'].map(lambda x: datetime.fromtimestamp(x.timestamp()))
        new_candle.set_index('time', inplace=True)
 
        df.iloc[-1] = new_candle.iloc[0]

        # Seleccionar solo las columnas necesarias para el gráfico de velas japonesas
        df_candlestick = df[['open', 'max', 'min', 'close']]
        df_candlestick.columns = ['Open', 'High', 'Low', 'Close']  # Cambiar el nombre de las columnas para mplfinance

        # Crear df_mpf
        df_mpf = df_candlestick.copy()
        df_mpf['num_index'] = range(len(df_mpf))

        # Calcular la EMA de 70 períodos
        ema_70 = df['close'].ewm(span=14).mean()
        df_mpf['EMA60'] = ema_70
        
        # Limpiar el eje antes de dibujar el nuevo gráfico.
        ax.clear()         

        # Dibujar las velas.    
    
        plot_colored_candles(df, ax, trendlines)
    
        ax.set_ylabel('precio')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
        
        # Agrega una cuadrícula en el eje X
        ax.xaxis.grid(True, linestyle='--', linewidth=0.5)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
    
        plt.gca().yaxis.set_tick_params(labelsize=7)
    
        # puntero al precio
        if price_line is not None:
            price_line.remove()
        last_price = df['close'].iloc[-1]

        price_line = ax.axhline(y=last_price, color='blue', linewidth=1, linestyle='--')
        
        # Agregar etiqueta de precio al lado izquierdo de la ventana
        last_price = df['close'].iloc[-1]
        if price_text is not None:
            price_text.remove()
        xlim = ax.get_xlim()
        left_x = xlim[0] + 2
        price_text = ax.annotate(f'{last_price:.6f}', xy=(left_x, last_price), xycoords='data', ha='right', va='center', fontsize=10, color='white', bbox=dict(facecolor='#0000FF', edgecolor='white', boxstyle='round,pad=0.2'))    

        # Configurar el título del gráfico en el objeto ax.    
        formatted_tolerance_levels = format(tolerance, '.6f')         
        tolerance = float(formatted_tolerance_levels)
        st_formated_level = str(formatted_tolerance_levels)
       
        # Graficar niveles a romper 3 toques
        for level_type, level_value in levels:
            ax.axhline(level_value, color='red' if level_type == 'support' else 'green', linestyle='--', linewidth=1.2)

        # Graficar niveles confirmados
        for level_type, level_value in levels_touched:
            ax.axhline(level_value, color='#4B0082' if level_type == 'support' else '#DA70D6', linestyle='--', linewidth=1.2)


        # Numerar todas las velas desde el inicio hasta el final
        for idx, row in df_mpf.iterrows():
           num_index = int(row['num_index'])
           high = row['High']
           color = "black"
        
           # Cambiar el color de los números de los índices de las velas según la tendencia
           for trend_type, m, b, start_index, end_index in trendlines:
               if start_index - 1 <= num_index <= end_index - 1:
                   color = "green" if trend_type == "ascending" else "red"
                   break

           ax.text(num_index, high, str(num_index + 1), ha='center', va='bottom', color=color, fontsize=8)
           

        #for line in ascending_trendlines:
        #    trend_type, m, b, start_index, end_index = line
        #    start_date = df_mpf.iloc[start_index].name
        #    end_date = df_mpf.iloc[end_index].name
        #    x_values = df_mpf.loc[start_date:end_date]['num_index']
        #    y_values = df_mpf.loc[start_date:end_date]['Low']
        #    ax.plot(x_values, y_values, color="green")

        #for line in descending_trendlines:
        #    trend_type, m, b, start_index, end_index = line
        #    start_date = df_mpf.iloc[start_index].name
        #    end_date = df_mpf.iloc[end_index].name
        #    x_values = df_mpf.loc[start_date:end_date]['num_index']
        #    y_values = df_mpf.loc[start_date:end_date]['High']
        #    ax.plot(x_values, y_values, color="red")


        # Graficar EMA de 60 períodos
        ax.plot(df_mpf['num_index'], df_mpf['EMA60'], color='goldenrod', linestyle='-', linewidth=1.3)
   
        # Actualizar y mostrar el gráfico
        plt.pause(0.2)
        
        # Esperar antes de actualizar el gráfico
        time.sleep(0.5)
        
        pass

    except KeyboardInterrupt:
           print("\n Interrupción de teclado detectada.")
           initial_investment, investment_operation, hibernate, time_to_hibernate, start_time, trade_done = menu(initial_investment, investment_operation)
           
        
        

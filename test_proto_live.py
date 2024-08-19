import pandas as pd
import numpy as np
import talib as ta
import datetime
import json
from colorama import Fore, Back, Style, init
import time
from scipy import stats
import sys
from datetime import datetime, timedelta
import matplotlib.animation as animation
import simpleaudio as sa
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.animation as animation
from iqoptionapi.stable_api import IQ_Option


init(autoreset=True)


tolerance = 0.00002
tolerance_trend = 0.00001

trades = []

# Pega aquí las funciones compartidas anteriormente
def find_levels(df, min_touches, tolerance):
    levels = []
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

            if support_level - tolerance <= df.iloc[i]['min'] <= support_level + tolerance:
                support_touches += 1
            if resistance_level - tolerance <= df.iloc[i]['max'] <= resistance_level + tolerance:
                resistance_touches += 1

            if support_touches >= min_touches and resistance_touches >= min_touches:
                break

        current_price = df.iloc[-1]['close']

        # Agregar niveles rotos a la lista de broken_levels
        if support_touches >= min_touches and df.iloc[index]['min'] < support_level - tolerance:
            broken_levels.append(('support', support_level))
        if resistance_touches >= min_touches and df.iloc[index]['max'] > resistance_level + tolerance:
            broken_levels.append(('resistance', resistance_level))

        # Agregar soportes por debajo del precio actual y resistencias por encima del precio actual
        if support_touches >= min_touches and support_level < current_price and ('support', support_level) not in levels:
            levels.append(('support', support_level))
        if resistance_touches >= min_touches and resistance_level > current_price and ('resistance', resistance_level) not in levels:
            levels.append(('resistance', resistance_level))

    return levels


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



def is_trendline(candles, trend_type, min_touches=3, tolerance_trend=0.0001):
    if len(candles) < min_touches:
        return False

    if trend_type == "ascending":
        # Verificar si todas las velas son alcistas
        for _, c in candles.iterrows():
            if c['close'] <= c['open']:
                return False
        corners = [(i, c['min']) for i, (_, c) in enumerate(candles.iterrows())]

    elif trend_type == "descending":
        # Verificar si todas las velas son bajistas
        for _, c in candles.iterrows():
            if c['close'] >= c['open']:
                return False
        corners = [(i, c['max']) for i, (_, c) in enumerate(candles.iterrows())]

    # Calcular la pendiente y la intersección
    m, b = np.polyfit([p[0] for p in corners], [p[1] for p in corners], 1)

    # Verificar si la pendiente es positiva (alcista) o negativa (bajista)
    if (trend_type == "ascending" and m <= 0) or (trend_type == "descending" and m >= 0):
        return False

    # Comprobar si todos los puntos tocan la línea de tendencia
    for idx, (_, c) in enumerate(candles.iterrows()):
        price = c['min'] if trend_type == "ascending" else c['max']
        if abs(price - (m * idx + b)) > tolerance_trend:
            return False

    return True

def find_trendlines(df, trend_type="ascending", min_touches=3, tolerance=tolerance_trend):
    trendlines = []

    for start_index in range(len(df) - min_touches + 1):
        for end_index in range(start_index + min_touches - 1, len(df)):
            candles = df.iloc[start_index:end_index + 1]

            if is_trendline(candles, trend_type, min_touches):
                # Calcular la pendiente y la intersección de la línea de tendencia
                if trend_type == "ascending":
                    corners = [(i, c['min'] if c['close'] > c['open'] else c['close']) for i, (_, c) in enumerate(candles.iterrows())]
                elif trend_type == "descending":
                    corners = [(i, c['max'] if c['close'] < c['open'] else c['close']) for i, (_, c) in enumerate(candles.iterrows())]

                m, b = np.polyfit([p[0] for p in corners], [p[1] for p in corners], 1)

                trendlines.append((trend_type, m, b, start_index, end_index))

    return trendlines

def is_touching_trendline(df, trend_type, current_candle_index, tolerance_trend=tolerance_trend):
    trendlines = find_trendlines(df, trend_type, min_touches=3, tolerance=tolerance_trend)
    
    for trendline in trendlines:
        _, m, b, start_index, end_index = trendline
        if start_index <= current_candle_index <= end_index:
            current_candle = df.iloc[current_candle_index]
            if trend_type == "ascending":
                price = current_candle['min']
            elif trend_type == "descending":
                price = current_candle['max']

            # Comprobar si el precio de la vela actual está dentro de la tolerancia de la línea de tendencia
            if abs(price - (m * (current_candle_index - start_index) + b)) <= tolerance_trend:
                return True
    return False


def record_trade(trades_list):
    wins = trades_list.count("win")
    losses = trades_list.count("loss")
    total_trades = len(trades_list)

    win_percentage = (wins / total_trades) * 100
    loss_percentage = (losses / total_trades) * 100

    return win_percentage, loss_percentage, total_trades


   
def plot_chart(df, levels, trendlines, df_mpf):
    fig, ax = plt.subplots()

    # Establecer el estilo del gráfico
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle=":", y_on_right=True)    

    return fig, ax, s


# Configura tus credenciales de IQOption aquí
email = ""
password = ""

# Conéctate a IQOption
def connect_to_iqoption():
    api = IQ_Option(email, password)
    connected = False
    while not connected:
        try:
            api.connect()
            if api.check_connect():
                connected = True
                print("Conectado exitosamente")
            else:
                print("Error al conectar")
                time.sleep(1)
        except Exception as e:
            print(f"Error al conectar: {e}")
            time.sleep(1)
    return api

api = connect_to_iqoption()
api.change_balance("PRACTICE")

# Parámetros
pair = 'EURUSD'
timeframe = 1  # en minutos

print("obteniendo datos...")


#fig, ax = plt.subplots()
fig, ax = plt.subplots(figsize=(13, 4))
plt.show(block=False)

mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
s = mpf.make_mpf_style(marketcolors=mc, gridstyle=":", y_on_right=True)

fig.canvas.manager.set_window_title('Accion del Precio por alex_strange')



#[ax.spines[side].set_visible(False) for side in ax.spines]

# Enabling minor grid lines:
ax.grid(which = "minor")
ax.minorticks_on()

# Get historical fresh data
candles_data = api.get_candles(pair, timeframe * 60, 120, time.time())
df = pd.DataFrame(candles_data)

df['at'] = pd.to_datetime(df['at'])
df.set_index('at', inplace=True)

df['time'] = df.index.map(lambda x: datetime.fromtimestamp(x.timestamp()))
df.set_index('time', inplace=True)

# Seleccionar solo las columnas necesarias para el gráfico de velas japonesas
df_candlestick = df[['open', 'max', 'min', 'close']]
df_candlestick.columns = ['Open', 'High', 'Low', 'Close']  # Cambiar el nombre de las columnas para mplfinance

# Crear df_mpf
df_mpf = df_candlestick.copy()
df_mpf['num_index'] = range(len(df_mpf))


# Calcula el ATR
atr = ta.ATR(df['max'], df['min'], df['close'], timeperiod=60)
atr = atr.dropna()
mean_atr = atr.mean()

# Valor de porcentaje mínimo
fix_percent_levels = 0.02  # 1 por ciento del valor de la media de atr
fix_percent_trends = 0.02  # 2 por ciento del valor de la media de atr


previous_candle_timestamp = None

status = False
valor = 0
id = 0
trade_done = False
while True:
    # Download the new candle
    new_candle_data = api.get_candles(pair, timeframe * 60, 1, time.time())  # Download 1 new candle
    new_df = pd.DataFrame(new_candle_data)

    # ...preprocessing steps for new_df...

    new_df_candlestick = new_df[['open', 'max', 'min', 'close']]
    new_df_candlestick.columns = ['Open', 'High', 'Low', 'Close']

    new_candle_timestamp = pd.to_datetime(new_df.at[0, 'at'])
    last_timestamp = df.index[-1]
    
    # Encuentra los niveles y líneas de tendencia
    levels = find_levels(df, min_touches=3, tolerance=tolerance)
    ascending_trendlines = find_trendlines(df, trend_type="ascending", min_touches=3, tolerance=tolerance_trend)
    descending_trendlines = find_trendlines(df, trend_type="descending", min_touches=3, tolerance=tolerance_trend)
    
    # Combina líneas de tendencia alcistas y bajistas
    trendlines = ascending_trendlines + descending_trendlines
    
    
    if trade_done:
            try:
                status, valor = api.check_win_digital_v2(id)
            except:
                status = True
                valor = 0
            if status:
                if valor > 0:
                   try:
                       w_object = sa.WaveObject.from_wave_file('money.wav')
                       p_object = w_object.play()
                       p_object.wait_done()
                   except FileNotFoundError:
                       print("Wav File does not exists")
                   trades.append("win")
                   win_percentage, loss_percentage, total_trades = record_trade(trades)
                   print(Fore.GREEN + " win " + Fore.WHITE + "operation, Profit: " + Fore.GREEN + f"{valor}" + Fore.RESET)
                   print(Fore.GREEN + " wins" + Fore.WHITE + f": {round(win_percentage,2)} %    " + Fore.RED + "loss" + Fore.WHITE + f": {round(loss_percentage,2)} %" + Fore.RESET)
                   trade_done = False
                else:
                   trades.append("loss")
                   win_percentage, loss_percentage, total_trades = record_trade(trades)
                   print(Fore.RED + " loss " + Fore.WHITE + "operation, Profit: " + Fore.RED + f"{valor}" + Fore.RESET)
                   print(Fore.GREEN + " wins" + Fore.WHITE + f": {round(win_percentage,2)} %    " + Fore.RED + "loss" + Fore.WHITE + f": {round(loss_percentage,2)} %" + Fore.RESET)
                   trade_done = False
    

    if new_candle_timestamp.minute != last_timestamp.minute:
        # A new candle has formed
        df = df.iloc[-119:]  # Discard the oldest candle
        df.loc[new_candle_timestamp] = new_df.iloc[0]  # Add the new candle
        # Graficar
        formatted_tolerance_levels = format(tolerance, '.6f')
        formatted_tolerance_trends = format(tolerance_trend, '.6f')
           
        tolerance = float(formatted_tolerance_levels)
        tolerance_trend  = float(formatted_tolerance_trends)
            
        st_formated_level = str(formatted_tolerance_levels)
        st_formated_trend = str(formatted_tolerance_trends)
           
        st_formated_level =st_formated_level[0:6] + Fore.CYAN + "[" + Fore.GREEN + st_formated_level[6:8] + Fore.CYAN + "]" + Fore.RESET
        st_formated_trend =st_formated_trend[0:6] + Fore.YELLOW + "[" + Fore.GREEN + st_formated_trend[6:8] + Fore.YELLOW + "]" + Fore.RESET
           
        print( " " + Fore.GREEN + pair + Fore.WHITE + " pip level tolerance: " + st_formated_level)
        print( " " + Fore.GREEN + pair + Fore.WHITE + " pip trend tolerance: " + st_formated_trend)
        
    else:
        # Update the last candle with the new data
        df.loc[last_timestamp] = new_df.iloc[0]

    # Verificar si estamos en una nueva vela
    current_candle_timestamp = df.index[-1]
    new_candle = current_candle_timestamp != previous_candle_timestamp
    previous_candle_timestamp = current_candle_timestamp

    current_time = datetime.now().time()
    seconds_remaining = 60 - current_time.second
    seconds_elapsed = current_time.second

    if seconds_remaining <= 1 or seconds_elapsed < 4:
        # Calcular las tendencias alcistas y bajistas
        current_candle_index = len(df) - 1
        current_candle = df.iloc[-1]
        previous_candle = df.iloc[-2]  # Agregar la vela anterior
        is_touching_ascending_trendline = is_touching_trendline(df, "ascending", current_candle_index)
        is_touching_descending_trendline = is_touching_trendline(df, "descending", current_candle_index)
        breaking_resistance, breaking_support = is_breaking_level(df, current_candle, previous_candle, levels, tolerance)

        # Imprimir la información sobre las tendencias solo si es el inicio de la vela
        if is_touching_ascending_trendline:
            print(Fore.GREEN + " CALL " + Fore.WHITE + "- microtendencia alcista" + Fore.RESET)
            direction = "call"
            status,id = api.buy_digital_spot(pair, 1, direction, 1)
            trade_done = True

        if is_touching_descending_trendline:
            print(Fore.RED + " PUT " + Fore.WHITE + "- microtendencia bajista" + Fore.RESET)
            direction = "put"
            status,id = api.buy_digital_spot(pair, 1, direction, 1)
            trade_done = True

        if breaking_resistance is not None:
            print(Fore.GREEN + " CALL " + Fore.WHITE + "- rompiendo nivel de resistencia" + Fore.RESET)
            direction = "call"
            status,id = api.buy_digital_spot(pair, 1, direction, 1)
            trade_done = True

        if breaking_support is not None:
            print(Fore.RED + " PUT " + Fore.WHITE + "- rompiendo nivel de soporte" + Fore.RESET)
            direction = "put"
            status,id = api.buy_digital_spot(pair, 1, direction, 1)
            trade_done = True


    # Crear df_mpf
    df_mpf = df[['open', 'max', 'min', 'close']].copy()
    df_mpf.columns = ['Open', 'High', 'Low', 'Close']  # Cambiar el nombre de las columnas para mplfinance
    df_mpf['num_index'] = range(len(df_mpf))

    # Calcula el ATR
    atr = ta.ATR(df['max'], df['min'], df['close'], timeperiod=60)
    atr = atr.dropna()
    mean_atr = atr.mean()

    # Valor de porcentaje mínimo
    fix_percent_levels = 0.02  # 1 por ciento del valor de la media de atr
    fix_percent_trends = 0.02  # 2 por ciento del valor de la media de atr

    # Calcula el porcentaje sobre la media del atr
    tolerance = mean_atr * fix_percent_levels
    tolerance_trend = mean_atr * fix_percent_trends


    # Limpiar el eje antes de dibujar el nuevo gráfico.
    ax.clear()

    # Dibujar las velas.
    #mpf.plot(df_mpf, type='candle', style=s, ax=ax, ylabel='Precio')
    
    mpf.plot(df_mpf, type='candle', style=s, ax=ax, update_width_config=dict(candle_linewidth=0.9, candle_width=0.6))
    ax.set_title(pair)
    ax.set_ylabel('Price')
    ax.xaxis.set_major_formatter(mdates.DateFormatter(''))
    
    # Configurar el título del gráfico en el objeto ax.
    ax.set_title('Niveles horizontales y líneas de tendencia')

    # Graficar niveles horizontales
    for level_type, level_value in levels:
        ax.axhline(level_value, color='red' if level_type == 'support' else 'green', linestyle='--', linewidth=1)


    # Numerar todas las velas desde el inicio hasta el final
    for idx, row in df_mpf.iterrows():
        num_index = int(row['num_index'])
        high = row['High']
        color = "black"
        
        # Cambiar el color de los números de los índices de las velas según la tendencia
        for trend_type, m, b, start_index, end_index in trendlines:
            if start_index <= num_index <= end_index:
                color = "green" if trend_type == "ascending" else "red"
                break

        ax.text(num_index, high, str(num_index + 1), ha='center', va='bottom', color=color, fontsize=8)


    #for trendline in trendlines:
    #    ax.plot(df_mpf.index, trendline[0] * df_mpf['num_index'] + trendline[1], color='orange', linestyle='--', linewidth=0.7)
    
    # Actualizar y mostrar el gráfico
    plt.pause(0.2)

    # Esperar antes de actualizar el gráfico
    time.sleep(0.2)    

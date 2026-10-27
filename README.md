# SAFEBOX AI

Dashboard biométrico local para una demo universitaria. Usa YuNet para detectar rostros y SFace para reconocer identidades con inferencia local en CPU. Exige reconocimiento facial y PIN numérico por usuario antes de abrir; también distingue escaneo, verificación, PIN pendiente, autorización, denegación y cámara desconectada. Incluye vista de cámara, similitud coseno real, cerradura y LED virtuales, historial y capturas. La interfaz oscura se organiza en **Inicio**, **Escáner**, **Usuarios**, **Historial** y **Configuración**.

## Instalar y ejecutar

En PowerShell, desde `C:\Users\USER\Desktop\Arduino`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_models.py
.\.venv\Scripts\python.exe app.py --self-test
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe app.py
```

Si ya existe `.venv`, instala los modelos una vez y ejecuta el último comando. Son los modelos ONNX oficiales de [OpenCV Zoo: YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) y [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface), usando la variante SFace INT8 para CPU (unos 10 MB en total). Se guardan en `models/`, excluido de Git. Si faltan, la interfaz abre y muestra el error; puedes descargarlos con el comando anterior y reiniciar. Concede permiso de cámara a las aplicaciones de escritorio en Windows.

## Uso

1. En el primer inicio, crea un **PIN de administrador** de 6 a 12 dígitos. Se guarda con sal y PBKDF2; cinco fallos bloquean el PIN durante 60 segundos. La sesión verificada dura dos minutos. Puedes cambiar el PIN o cerrar la sesión en **Configuración**.
2. En **Inicio** o **Usuarios**, pulsa **Registrar rostro** e introduce el PIN de administrador cuando se solicite. En la ventana, escribe un nombre y crea un **PIN de acceso propio de ese usuario**, de 6 a 12 dígitos. Mantén una sola cara frente a la cámara hasta completar las 20 muestras.
3. Vuelve a aparecer frente a la cámara y mantén el rostro visible. Tras cuatro lecturas consistentes, introduce el PIN de acceso en un plazo de 15 segundos. **Solo rostro + PIN correctos** permiten ordenar `OPEN`; una posible coincidencia facial por sí sola no autoriza. Al terminar el tiempo configurado, pasa a **ESPERANDO CIERRE**. Confirma manualmente que la puerta está cerrada y pulsa **Confirmar cierre** para ordenar `LOCK`. Sin sensor, esa confirmación es del operador y no prueba el cierre físico.
4. Un rostro no registrado o varios rostros producen **DENEGADO**. Si la cámara falla, usa **Reintentar cámara**. Los errores de comunicación dejan la cerradura en **ERROR** e impiden nuevas aperturas hasta reconectar y confirmar el cierre.
5. En **Usuarios** puedes asignar o cambiar el PIN de acceso de un perfil existente mediante **PIN acceso**, tras verificar el PIN de administrador. Los usuarios anteriores se conservan, pero **no podrán abrir hasta tener PIN de acceso**. Los registros LBPH previos requieren nuevo enrolamiento SFace. También puedes eliminar usuarios con confirmación. En **Historial** puedes buscar, filtrar, abrir capturas y exportar CSV sin fotografías ni vectores biométricos.

Cada intento resuelto se registra una vez mientras el rostro permanece visible. El historial se guarda en `datos/historial.db`, las capturas en `datos/capturas/`, los perfiles SFace en `datos/rasgos_sface/` y el índice de usuarios en `datos/usuarios.json`. Las muestras LBPH anteriores de `datos/rostros/` se conservan. Estos archivos contienen datos biométricos: no los publiques en repositorios.

## Configuración, simulación y ESP32

El modo predeterminado es `simulacion`. En **Configuración** puedes cambiar el umbral SFace, el índice de cámara (0–9), el tiempo de autorización (3–60 s), el modo y el puerto `COMx`; se guardan en `datos/config.json`. Los argumentos `--mode`, `--port`, `--similarity-threshold`, `--camera-index` y `--authorization-seconds` tienen prioridad en ese inicio. Para hardware instala la dependencia opcional:

```powershell
.\.venv\Scripts\python.exe -m pip install pyserial
.\.venv\Scripts\python.exe app.py --mode hardware --port COM4
```

El firmware combinado que usa la aplicación está en `firmware/SafeBoxLEDsESP32/SafeBoxLEDsESP32.ino` para Arduino-ESP32 3.x. Conecta la señal MG90S a **GPIO 18**, el LED azul a **GPIO 25** y el rojo a **GPIO 26**, cada LED con resistencia adecuada. Alimenta el servo desde una fuente externa regulada de **5 V** y conecta su GND al GND de la ESP32; no lo alimentes desde un GPIO. `firmware/SafeBoxLEDsESP32/calibration.h` está temporalmente en **DEMO sin puerta**: `OPEN=2000 µs` mueve el brazo a la izquierda y `LOCK=1500 µs` lo devuelve al centro, según el recorrido observado sin carga. `SERVO_STATUS` responde `DEMO`; estos valores **no certifican un pestillo bloqueado o liberado**. El inicio, reinicio y reconexión no ordenan movimientos. `python app.py` y **Run** de VS Code inician ahora en hardware DEMO por **COM4**. Confirma manualmente el retorno al centro antes de autenticar rostro + PIN. Para usar solo simulación, ejecuta `python app.py --mode simulacion --led-port COM4`.

En **Configuración**, con el modo hardware DEMO conectado, los campos **Autorización · µs** y **Retorno · µs** muestran los valores activos de la ESP32. **Guardar recorrido MG90S** exige PIN de administrador, valida dos valores distintos entre 1000 y 2000 µs y los guarda en la memoria local de la ESP32 con ACK. Guardar no mueve el motor; los cambios persisten tras reiniciar. No ajustes estos pulsos con una puerta acoplada sin comprobar antes el recorrido libre. Los valores de la cabecera son el punto de partida si la ESP32 aún no tiene una configuración guardada.

### Calibración inicial del MG90S

1. Deja el servo **sin acoplar a la puerta ni al pestillo**; si es posible, retira el brazo durante la primera prueba. La puerta deslizante de cartón reforzado (capas de 1,5 mm) se moverá manualmente: el servo solo debe liberar o bloquear, nunca arrastrarla.
2. En **Arduino IDE**, abre `firmware/SafeBoxLEDsESP32/SafeBoxLEDsESP32.ino`, elige **ESP32 Dev Module** (`esp32:esp32:esp32`), puerto **COM4**, y verifica/carga el sketch. Cierra SAFEBOX AI y el Monitor Serial antes de cargar, porque COM4 solo admite un proceso a la vez.
3. Abre el Monitor Serial a **115200 baudios** con fin de línea **Nueva línea**. Envía `PING 1` y copia el `ID_ARRANQUE` de `ACK 1 PING ID_ARRANQUE`. Envía `SERVO_STATUS 2 ID_ARRANQUE`: con la demo actual responde `ACK 2 SERVO_STATUS ID_ARRANQUE DEMO`. Sustituye `ID_ARRANQUE` por el valor real, sin escribir esa palabra literalmente.
4. Para una **nueva calibración manual**, deja el servo sin carga, pon temporalmente ambos pulsos de `calibration.h` en `0` y `SAFEBOX_DEMO_ONLY` en `0`, y vuelve a cargar el sketch. Envía **solo cuando lo decidas** `TEST_SERVO 3 ID_ARRANQUE 1500`. El firmware admite 1000–2000 µs, exige empezar en 1500 µs y después cambios de **50 µs o menos**. Avanza hacia los extremos solo mientras el eje gire libremente; estos son límites de pulso de prueba, no posiciones mecánicas verificadas. La señal PWM se detiene sola tras 600 ms; `SERVO_STOP 4 ID_ARRANQUE` la detiene de inmediato. También puedes usar **Configuración → Probar MG90S** en la aplicación, tras autenticar al administrador. El ACK solo confirma que el ESP32 aceptó PWM; no confirma movimiento físico. No fuerces el brazo.
5. Cuando exista el pestillo real, mide los pulsos que **realmente** lo liberan y bloquean. Escríbelos en `calibration.h` como `SAFEBOX_OPEN_US` y `SAFEBOX_LOCK_US`, distintos y entre 1000 y 2000 µs, y cambia `SAFEBOX_DEMO_ONLY` a `0`. Recompila y carga el sketch: `SERVO_STATUS` debe responder `LISTO`. No hay posiciones de puerta verificadas todavía.

El protocolo serial usa 115200 baudios. `PING SECUENCIA` recibe `ACK SECUENCIA PING ID_ARRANQUE`; los demás comandos usan `COMANDO SECUENCIA ID_ARRANQUE` y reciben `ACK` o `ERR` con esos datos (`TEST_SERVO` añade el pulso; `SERVO_STATUS` añade el estado al ACK). El ID cambia al reiniciar la ESP32 y el firmware rechaza comandos con el ID anterior. `AUTH_OK` enciende solo el azul durante 3 segundos; `AUTH_FAIL` mantiene el rojo hasta `LED_OFF`; ninguno mueve el servo. Solo `OPEN` y `LOCK` mueven el servo en modo DEMO o, tras calibración real, el pestillo. La aplicación envía `PING` aproximadamente cada 5 s y pasa a **ERROR / estado físico desconocido** ante desconexión, reinicio, ACK incorrecto o timeout. El ACK confirma la aceptación del comando, **no** la posición del servo ni el cierre de la puerta. Conectar o reconectar solo consulta estado; después debes confirmar manualmente el cierre antes de ordenar `LOCK`. Si pasas a simulación, la interfaz mantiene explícito que el estado físico es desconocido. `pyserial` se necesita para comunicación serial.

## Alcance técnico

El reconocimiento compara un vector SFace con el perfil de cada usuario y muestra la **similitud coseno real**: un valor mayor indica mayor similitud. El umbral predeterminado y mínimo de seguridad es `0.500`; las preferencias anteriores más bajas se elevan automáticamente en memoria. Puedes ajustarlo entre `0.500` y `0.900` según pruebas con personas autorizadas y desconocidas. Con varios perfiles se exige además separación entre la mejor y la segunda coincidencia. Estas medidas reducen falsos positivos, pero no garantizan eliminarlos. El PIN de acceso se almacena con sal y PBKDF2, con bloqueo temporal tras cinco errores; se guarda en `datos/access_pins.json`, excluido de Git. El sistema no incorpora prueba de vida y una fotografía podría engañar al reconocimiento, por lo que no debe usarse como único control de una caja fuerte real.

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
.\.venv\Scripts\python.exe app.py --mode hardware --port COM3
```

El firmware está en `firmware/SafeBoxESP32/SafeBoxESP32.ino`. Cárgalo con Arduino IDE y soporte **Arduino-ESP32 3.x** para la placa WeMos D1 ESP32-WROOM. Valores iniciales: servo MG90S en GPIO 18, LED azul en 25 y LED rojo en 26; cambia los `#define` antes de compilar si tu cableado difiere. Usa alimentación externa adecuada de 5 V para el servo y masa común con la ESP32; no alimentes el servo desde un GPIO. Ajusta `SAFEBOX_LOCK_US` y `SAFEBOX_OPEN_US` al mecanismo real antes de acoplarlo: el firmware limita ambos pulsos a 1000–2000 µs y **no mueve el servo al arrancar**. El montaje físico aún no se ha probado.

El protocolo serial usa 115200 baudios y líneas `COMANDO SECUENCIA`: `PING`, `OPEN`, `LOCK`, `DENIED` o `IDLE`; la ESP32 responde `ACK SECUENCIA COMANDO` o `ERR SECUENCIA COMANDO`. El ACK confirma la aceptación del comando, **no** el movimiento ni el estado físico de la puerta. Tras conectar en hardware, confirma el cierre desde la aplicación para enviar el primer `LOCK`. En error serial se bloquean nuevas aperturas. `pyserial` solo se necesita en modo hardware.

## Alcance técnico

El reconocimiento compara un vector SFace con el perfil de cada usuario y muestra la **similitud coseno real**: un valor mayor indica mayor similitud. El umbral predeterminado y mínimo de seguridad es `0.500`; las preferencias anteriores más bajas se elevan automáticamente en memoria. Puedes ajustarlo entre `0.500` y `0.900` según pruebas con personas autorizadas y desconocidas. Con varios perfiles se exige además separación entre la mejor y la segunda coincidencia. Estas medidas reducen falsos positivos, pero no garantizan eliminarlos. El PIN de acceso se almacena con sal y PBKDF2, con bloqueo temporal tras cinco errores; se guarda en `datos/access_pins.json`, excluido de Git. El sistema no incorpora prueba de vida y una fotografía podría engañar al reconocimiento, por lo que no debe usarse como único control de una caja fuerte real.

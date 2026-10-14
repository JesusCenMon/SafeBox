# SAFEBOX AI

Dashboard biométrico local para una demo universitaria. Usa YuNet para detectar rostros y SFace para reconocer identidades con inferencia local en CPU. Muestra cinco estados: **escaneando**, **verificando**, **autorizado**, **denegado** y **cámara desconectada**. Incluye vista de cámara, similitud coseno real, cerradura y LED virtuales, historial y capturas. La interfaz oscura se organiza en **Inicio**, **Escáner**, **Usuarios**, **Historial** y **Configuración**.

## Instalar y ejecutar

En PowerShell, desde `C:\Users\USER\Desktop\Arduino`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_models.py
.\.venv\Scripts\python.exe app.py --self-test
.\.venv\Scripts\python.exe app.py
```

Si ya existe `.venv`, instala los modelos una vez y ejecuta el último comando. Son los modelos ONNX oficiales de [OpenCV Zoo: YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) y [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface), usando la variante SFace INT8 para CPU (unos 10 MB en total). Se guardan en `models/`, excluido de Git. Si faltan, la interfaz abre y muestra el error; puedes descargarlos con el comando anterior y reiniciar. Concede permiso de cámara a las aplicaciones de escritorio en Windows.

## Uso

1. En **Inicio** o **Usuarios**, pulsa **Registrar rostro**. En la ventana, escribe un nombre e inicia la captura. Mantén una sola cara frente a la cámara hasta completar las 20 muestras. Puedes cancelar el registro.
2. Vuelve a aparecer frente a la cámara. Tras tres lecturas consistentes, el sistema muestra **AUTORIZADO**, la cerradura **DESBLOQUEADA** y el LED **AZUL**.
3. Un rostro no registrado o varios rostros producen **DENEGADO**, cerradura **BLOQUEADA** y LED **ROJO**. Al no haber rostro, vuelve a **ESCANEANDO** con la cerradura bloqueada. Si la cámara falla, usa **Reintentar cámara**.
4. En **Usuarios** puedes eliminar un registro tras confirmar la acción. Los registros LBPH previos aparecen como **requiere nuevo enrolamiento**: pulsa **Reenrolar** con el mismo nombre para crear su perfil SFace sin borrar las muestras antiguas. En **Historial** puedes buscar, filtrar, cambiar de página y abrir las capturas. Navegar no reinicia la cámara.

Cada intento resuelto se registra una vez mientras el rostro permanece visible. El historial se guarda en `datos/historial.db`, las capturas en `datos/capturas/`, los perfiles SFace en `datos/rasgos_sface/` y el índice de usuarios en `datos/usuarios.json`. Las muestras LBPH anteriores de `datos/rostros/` se conservan. Estos archivos contienen datos biométricos: no los publiques en repositorios.

## Simulación y futura WeMos

El modo predeterminado es `simulacion`: muestra y actualiza los estados sin hardware. El modo `hardware` está preparado para enviar líneas ASCII terminadas en salto de línea a una WeMos por serial: `OPEN`, `LOCK` y `DENIED`.

```powershell
.\.venv\Scripts\python.exe -m pip install pyserial
.\.venv\Scripts\python.exe app.py --mode hardware --port COM3
```

`pyserial` es opcional y solo se necesita para ese modo. El firmware de la WeMos debe implementar esos tres comandos, la velocidad de **115200 baudios** y una posición bloqueada segura al arrancar. Aún no se incluye firmware ni se ha probado una cerradura física.

## Alcance técnico

El reconocimiento compara un vector SFace con el perfil de cada usuario y muestra la **similitud coseno real**: un valor mayor indica mayor similitud. El umbral predeterminado es `0.363`; puedes ajustarlo al iniciar, por ejemplo `python app.py --similarity-threshold 0.45`. Este valor no es un porcentaje ni una probabilidad. El historial LBPH anterior conserva su distancia etiquetada como tal. El sistema no incorpora prueba de vida y una fotografía podría engañarlo, por lo que no debe ser el único control de una caja fuerte real.

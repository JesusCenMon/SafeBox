# SAFEBOX AI

Dashboard biométrico local para una demo universitaria. Usa la cámara del portátil para registrar y reconocer rostros autorizados. Muestra cinco estados: **escaneando**, **verificando**, **autorizado**, **denegado** y **cámara desconectada**. Incluye vista de cámara con marco biométrico animado, identidad, distancia LBPH real, cerradura y LED virtuales, historial y capturas de los intentos.

## Instalar y ejecutar

En PowerShell, desde `C:\Users\USER\Desktop\Arduino`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py --self-test
.\.venv\Scripts\python.exe app.py
```

Si ya existe `.venv`, basta el último comando. Concede permiso de cámara a las aplicaciones de escritorio en Windows y cierra cualquier otra aplicación que la esté usando.

## Uso

1. Pulsa **Registrar nuevo usuario**. En la ventana, escribe un nombre e inicia la captura. Mantén una sola cara frente a la cámara hasta completar las 20 muestras. Puedes cancelar el registro.
2. Vuelve a aparecer frente a la cámara. Tras tres lecturas consistentes, el sistema muestra **AUTORIZADO**, la cerradura **DESBLOQUEADA** y el LED **AZUL**.
3. Un rostro no registrado o varios rostros producen **DENEGADO**, cerradura **BLOQUEADA** y LED **ROJO**. Al no haber rostro, vuelve a **ESCANEANDO** con la cerradura bloqueada. Si la cámara falla, usa **Reintentar cámara**.

Cada intento resuelto se registra una vez mientras el rostro permanece visible. La lista de la interfaz muestra los accesos recientes. El historial completo se guarda en `datos/historial.db`, las capturas en `datos/capturas/` y los rostros registrados en `datos/rostros/` con sus nombres en `datos/usuarios.json`. Estos archivos contienen datos biométricos: no los publiques en repositorios. Para reiniciar todos los registros, cierra la app y elimina `datos/`.

## Simulación y futura WeMos

El modo predeterminado es `simulacion`: muestra y actualiza los estados sin hardware. El modo `hardware` está preparado para enviar líneas ASCII terminadas en salto de línea a una WeMos por serial: `OPEN`, `LOCK` y `DENIED`.

```powershell
.\.venv\Scripts\python.exe -m pip install pyserial
.\.venv\Scripts\python.exe app.py --mode hardware --port COM3
```

`pyserial` es opcional y solo se necesita para ese modo. El firmware de la WeMos debe implementar esos tres comandos, la velocidad de **115200 baudios** y una posición bloqueada segura al arrancar. Aún no se incluye firmware ni se ha probado una cerradura física.

## Alcance técnico

La detección usa Haar y el reconocimiento usa LBPH de OpenCV Contrib, compatible con los rostros registrados en la versión anterior. Se conserva esta ruta para evitar descargar modelos adicionales. La interfaz muestra la **distancia LBPH real**: un valor menor indica mayor similitud; se autoriza cuando es igual o inferior a `MAX_DISTANCE` en `core.py`. No es un porcentaje ni una probabilidad. El historial anterior se conserva; las entradas nuevas guardan la distancia y las antiguas pueden mostrarla vacía. El umbral puede requerir ajuste según cámara e iluminación. El sistema no incorpora prueba de vida y una fotografía podría engañarlo, por lo que no debe ser el único control de una caja fuerte real.

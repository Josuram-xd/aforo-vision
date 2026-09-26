# PRD — aforo-vision

**Repositorio:** `aforo-vision`
**Última actualización:** 26 de septiembre de 2026

## 1. Problema

En la Universidad Cooperativa de Colombia (campus Pasto) no hay ninguna forma automatizada de saber quién entra y quién sale de un salón durante una clase. El proyecto original apuntaba a la entrada principal del campus (un corredor de 4 partes), pero el profesor pidió simplificar: **una prueba piloto de un solo día, en la puerta de un solo salón, para un curso específico.**

`aforo-vision` es el componente que resuelve la parte de visión por computador de ese piloto: ve la puerta por 2 cámaras, decide si alguien entró o salió, y trata de identificar a los estudiantes del curso que ya fueron enrolados.

## 2. Objetivo

Durante la sesión de un curso, en una sola puerta:

1. Detectar cada persona que cruza la puerta.
2. Determinar si fue una **entrada** o una **salida**.
3. Identificar, cuando sea posible, a cuál estudiante enrolado corresponde (por reconocimiento periocular — funciona con tapabocas).
4. Si no se puede identificar por rostro, seguir contando a la persona (por re-identificación de cuerpo) para que el aforo no falle.
5. Resistir los dos trucos más obvios: caminar de espaldas para no ser reconocido, y que una persona se tape detrás de otra.
6. Mandar cada evento resuelto (nunca video) al backend en la nube.

## 3. No-objetivos (fuera de alcance para este piloto)

- No es una instalación permanente — corre solo el día de la prueba, en el laptop de Josuram, con supervisión.
- No cubre el corredor de 4 partes del campus (torniquetes + pedestales) — eso queda documentado como arquitectura objetivo a futuro, no como parte de este piloto.
- No entrena modelos de reconocimiento facial desde cero — usa modelos pre-entrenados (transfer learning / embeddings ya entrenados).
- No procesa ni transmite video a la nube — todo el procesamiento es local.
- No maneja autenticación de usuarios ni permisos — es un piloto de un día, supervisado.

## 4. Usuarios

- **El profesor**, que evalúa la demo y los resultados del piloto.
- **Josuram**, que opera el laptop y las cámaras el día de la prueba.
- **Los estudiantes del curso**, que son las personas detectadas (algunos enrolados con consentimiento, identificables por nombre; el resto solo contados).

## 5. Alcance del piloto (estado actual, 26 sept 2026)

| Aspecto | Decisión |
|---|---|
| Ubicación | Una sola puerta de un salón de clase |
| Cámaras | 2 — una **USB** (cerca del laptop, adentro del salón) y una **WiFi** (afuera, en el pasillo, más lejos del laptop) |
| Cómputo | El laptop personal de Josuram, solo el día del piloto (no queda instalado ni corriendo sin supervisión) |
| Personas identificables | Los estudiantes del curso que dieron consentimiento informado — puede ser más o menos de 5, según el roster real del curso |
| Dirección de cruce | Por orden de checkpoint: si a alguien lo detecta primero la cámara de **afuera** y luego la de **adentro** → entrada; al revés → salida |
| Red | Se recomienda un hotspot personal (celular) para conectar la cámara WiFi con el laptop, para no depender del WiFi institucional (que suele aislar dispositivos entre sí) |

## 6. Requisitos funcionales

1. Capturar video de ambas cámaras (USB e IP/WiFi) con la misma interfaz de código.
2. Detectar personas y su pose corporal en cada frame.
3. Trackear cada persona detectada a través de los frames (por cámara).
4. Intentar identificar el rostro por la región periocular contra la lista de estudiantes enrolados.
5. Si el rostro no es identificable, extraer un embedding de apariencia corporal para no perder el conteo.
6. Emparejar las detecciones de la cámara de afuera con las de la cámara de adentro (misma persona, ventana de pocos segundos) para determinar la dirección.
7. Evitar contar dos veces el mismo cruce.
8. Manejar el caso de 2 personas que se tapan una a otra (oclusión parcial).
9. Construir el evento resuelto (JSON) y enviarlo al backend (`aforo-backend`) por HTTPS.
10. Si falla el envío (red caída), guardar el evento en una cola local y reintentar cuando vuelva la conexión.
11. (Opcional, para la demo) Exponer una vista de video anotado en la red local, para mostrar en vivo cómo el sistema está detectando.

## 7. Métricas de éxito del piloto

- El conteo de entradas/salidas coincide con un conteo manual de control durante la sesión.
- Al menos una captura clara por estudiante enrolado durante la sesión (para poder evaluar la tasa de identificación real).
- Cero eventos duplicados por el mismo cruce.
- Ningún evento se pierde si hay una caída de red breve.

## 8. Restricciones

- Corre en el laptop personal de Josuram — debe funcionar sin GPU dedicada.
- Python 3.14.
- El video nunca sale del laptop; solo el JSON del evento resuelto.
- Presupuesto de hardware para las cámaras: dentro de los ~500.000 COP totales del proyecto (compartido con las otras piezas de hardware).

## 9. Preguntas abiertas

- Fecha exacta del piloto y salón asignado.
- Roster final de estudiantes que dan consentimiento informado antes del día de la prueba.

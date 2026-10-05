# Autorización informada para el tratamiento de datos biométricos

> **BORRADOR / PLANTILLA.** Los textos entre `[corchetes]` se completan antes de usarla. Antes de pedirle a alguien que firme, debe revisarla la oficina de protección de datos o asesoría jurídica de la universidad: este documento no es asesoría legal. Marco de referencia: Ley 1581 de 2012 y su reglamentación (Decreto 1074 de 2015, capítulo 25).
>
> Los formularios firmados contienen datos personales: **no se suben al repositorio** ni a ningún servicio en la nube. Se guardan en físico o en una carpeta local protegida.

**Proyecto:** Prueba piloto de conteo de aforo en la puerta de un salón (`aforo-vision`).
**Fecha de la prueba:** [fecha del piloto] — **Lugar:** [salón y sede, Universidad Cooperativa de Colombia, campus Pasto]

## 1. Quién trata sus datos

| | |
|---|---|
| Responsable del tratamiento | [nombre de la persona o entidad responsable; confirmar con la universidad si es el estudiante a cargo del piloto, el docente o la institución] |
| Contacto para consultas y reclamos | [correo] — [teléfono, opcional] |
| Docente que supervisa el piloto | [nombre del docente] |

## 2. Para qué se usan sus datos (finalidad)

Únicamente para esta prueba piloto de **un solo día**: saber cuántas personas hay en el salón y reconocer, por su nombre, a los estudiantes que aceptan participar, para evaluar si el sistema funciona. Sus datos **no se usarán** para evaluarlo académicamente, controlar su asistencia a efectos de notas, ni para ningún otro fin.

## 3. Qué datos se tratan y cómo

| Dato | Qué es | Dónde queda |
|---|---|---|
| Nombre | El que usted indica al inscribirse. | En el laptop del proyecto y en la base de datos en la nube del piloto (AWS), junto con su estado (adentro/afuera). |
| **Huella facial de la zona de los ojos** (dato biométrico, **sensible**) | Una lista de números calculada a partir de la región de los ojos de su rostro, que sirve para reconocerle aunque lleve tapabocas. **No es una fotografía** y no permite reconstruir su cara a simple vista. | **Solo en el laptop** del proyecto, en una carpeta local. No se sube a la nube. |
| Eventos de cruce | Un registro por cada entrada o salida: identificador anónimo (código aleatorio), tipo de evento, hora y si se le reconoció por el rostro. | En la base de datos en la nube del piloto. |
| Video de las cámaras | Las cámaras de la puerta se procesan en tiempo real. | **No se graba y no sale del laptop.** Si se muestra en vivo durante una demostración, es solo en la red local y solo a personas con usuario y contraseña. |

Si usted **no** se inscribe, el sistema igualmente cuenta su paso por la puerta usando la apariencia general del cuerpo, pero **no** lo identifica ni le asocia ningún nombre. Esa información [se calcula en memoria y se descarta; confirmar antes del piloto que así funciona la versión final].

El sistema se basa en modelos estándar de reconocimiento facial (ArcFace/InsightFace). No existe un modelo "periocular" propio ni comercial; solo se recorta la zona de los ojos y se procesa con ese modelo.

**Posible transferencia internacional:** la base de datos en la nube está alojada en Amazon Web Services, región [región, p. ej. us-east-1], fuera de Colombia. Allí solo van su nombre, su estado y sus eventos, **nunca** su huella facial ni video. [Confirmar con asesoría jurídica si esto requiere autorización adicional.]

## 4. Su decisión es voluntaria

- Los datos biométricos son **datos sensibles**: usted **no está obligado** a autorizar su tratamiento.
- Si no acepta, **no pasa nada**: no afecta su nota, su matrícula ni su relación con la universidad. [Indicar la alternativa para quien no participe, p. ej. puerta alterna, si aplica.]
- Puede aceptar solo una parte (por ejemplo, ser contado sin ser identificado por nombre).

## 5. Sus derechos

Como titular puede, en cualquier momento y sin costo:

- **Conocer** qué datos suyos se tratan y cómo se usan.
- **Actualizar y rectificar** sus datos.
- **Solicitar la supresión** de sus datos y **revocar** esta autorización.
- Pedir **prueba de esta autorización**.
- Presentar una queja ante la **Superintendencia de Industria y Comercio** si considera que se vulneraron sus derechos.

Para ejercerlos escriba a [correo de contacto]. Si revoca, **se borrará su huella facial del laptop y su perfil de la base de datos en la nube** en un plazo máximo de [plazo, p. ej. 5 días hábiles] y se le confirmará por escrito.

## 6. Cuánto tiempo se guardan

| Dato | Se elimina |
|---|---|
| Huella facial (laptop) | Al terminar el piloto, a más tardar el [fecha límite]. |
| Nombre, estado y eventos (nube) | Al terminar el piloto, a más tardar el [fecha límite]. [Si se conservan resultados agregados, que sean anónimos: sin nombres.] |
| Ensayo previo | Los datos del ensayo se borran antes del día de la prueba. |

## 7. Seguridad

La huella facial se guarda solo en el laptop del proyecto, fuera del repositorio de código, y no se comparte con terceros. El acceso al video en vivo, si se habilita, requiere usuario y contraseña. [Agregar medidas concretas: cifrado de disco, quién tiene acceso al laptop, etc.]

## 8. Autorización

Declaro que soy mayor de edad, que leí este documento, que se me explicaron la finalidad, los datos que se tratan, el carácter voluntario de mi decisión y mis derechos, y que:

- [ ] **Autorizo** el tratamiento de mi huella facial (zona de los ojos) para ser reconocido por nombre durante el piloto.
- [ ] **Autorizo** el registro de mi nombre y de mis entradas y salidas en la base de datos en la nube del piloto.
- [ ] **No autorizo** lo anterior (solo seré contado, sin nombre).

| | |
|---|---|
| Nombre completo | |
| Documento de identidad | |
| Correo de contacto | |
| Fecha | |
| Firma | |

*Para uso del proyecto:* `personId` asignado: ____________________ (lo genera `python -m enrollment.enroll_student`; es un UUID v4). **Enrolar solo a quien marcó la primera casilla.**

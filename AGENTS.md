# AGENTS.md — aforo-vision

Este archivo le dice a los asistentes de IA (Claude Code, Cursor, GitHub Copilot, etc.) cómo comportarse dentro de este repositorio.

## Qué es este repositorio

`aforo-vision` es el pipeline local de visión por computador del piloto de aforo. Corre en el laptop de Josuram, lee 2 cámaras (una USB, una WiFi), y produce eventos JSON (`ENTRY`/`EXIT` + identidad opcional) que envía por HTTPS a `aforo-backend`. **No** tiene base de datos ni interfaz visual — es un productor de eventos.

Antes de proponer cambios, lee `PRD.md` (qué debe hacer el sistema) y `ARCHITECTURE.md` (cómo está construido, en inglés).

> **Versión 2 en curso (oct 2026):** los eventos van a Supabase en vez del API de AWS, y se está evaluando correr la visión en un equipo del salón o en una VM gratis en la nube en vez del laptop (`ARCHITECTURE.md` sección 10, `TASKS.md` tasks 14-16). Las reglas marcadas **(v2)** aplican a ese trabajo.

## Reglas de git (obligatorias, sin excepciones)

1. **Una rama por task.** Los asistentes de IA pueden hacer commit y push, pero **solo en ramas de trabajo**, nunca en `main`. Nombre de la rama: `task-<n>-<descripcion-corta>` (ej. `task-7-supabase-schema`). Si la task depende de otra que aún no está en `main`, la rama sale de la rama de esa task (PRs encadenados).
2. **Un commit por subtarea**, con el mensaje exacto de `TASKS.md` y el sufijo `[x.y]`; en ese mismo commit se marca `[x]` en `TASKS.md`.
3. **Prohibido para la IA:** `git merge`, `git rebase`, `git reset --hard`, `git tag`, `push --force` y push a `main`; tampoco fusionar PRs. Puede **abrir** el PR de la rama hacia `main` (o hacia la rama de la que depende); la revisión y el merge los hace siempre una persona.
4. **Nadie puede tener `Co-Authored-By` de una IA.** Ningún commit ni PR puede incluir líneas `Co-Authored-By: Claude ...` (ni de ninguna otra IA), `noreply@anthropic.com` ni "Generated with Claude Code". El autor es el usuario configurado en git.
5. **Cómo se hace cumplir:**
   - `.githooks/commit-msg` rechaza localmente esos mensajes. Actívalo una vez por clon: `git config core.hooksPath .githooks`. No se usa `--no-verify`.
   - `.github/workflows/no-ai-coauthor.yml` falla en GitHub si algún commit del historial los tiene.
   - `.claude/settings.json` desactiva la coautoría automática de Claude Code y bloquea merge, rebase, force push y push a `main`.
   No desactives ni modifiques estos tres archivos sin que el usuario lo pida explícitamente.

## Reglas para el agente

1. **El video nunca sale de este repositorio ni del laptop.** Ningún cambio debe agregar código que suba frames, clips o imágenes crudas a un servicio externo (nube, Amplify, `aforo-backend`, servicios de streaming). Solo el JSON del evento resuelto viaja fuera (ver `ARCHITECTURE.md` sección 4.8-4.9). La única excepción es el servidor de video de la sección 4.10: transmite **solo en la red local**, **solo a usuarios con login** (Cognito, grupos `viewer`/`dev`) y nunca graba. **(v2)** Esta regla sigue vigente hasta que se decida el ADR-007 (task 15.2). Si se elige procesar en la nube, la única excepción será el túnel cifrado (Tailscale/WireGuard) hacia el servidor propio del equipo, procesando en memoria, sin grabar y sin servicios de terceros; y se actualiza esta regla y el aviso de videovigilancia antes de usarlo.
2. **No inventes un "modelo periocular" propietario.** La identificación periocular se hace recortando la región de los ojos y pasándola por un modelo de embeddings faciales estándar (ArcFace/InsightFace) — no existe un modelo comercial dedicado "periocular", y el código/documentación debe seguir siendo honesto sobre esto.
3. **Mantén ambas cámaras simétricas.** `camera-outside` y `camera-inside` deben correr la misma lógica de detección/tracking/identidad/anti-spoofing. No agregues lógica que sea exclusiva de una sola cámara sin actualizar también la otra y documentar por qué.
4. **La dirección (ENTRY/EXIT) se decide por orden de checkpoint**, no por orientación corporal. Si tocas `src/direction/resolver.py`, no reintroduzcas lógica de "mirando hacia adelante/atrás" como señal primaria — eso ya se descartó por ser vulnerable a caminar de espaldas (ver ADR-003 en `ARCHITECTURE.md`).
5. **Sigue el contrato de evento JSON compartido** definido en `ARCHITECTURE.md` (este repo) y `aforo-backend/ARCHITECTURE.md`. Si cambias un campo del evento, avisa explícitamente que también hay que actualizar `aforo-backend` y `aforo-frontend` — no lo hagas de forma aislada. **(v2)** El destino es la RPC `ingest_event` de Supabase (`aforo-db`), con el mismo JSON; cualquier cambio también se avisa a `aforo-db`.
6. **No agregues Kafka, MQTT, Docker o colas distribuidas.** Ya se decidió (ADR en `aforo-backend`) que esta pieza no las necesita — es una sola fuente de eventos, de baja frecuencia, para un piloto de un día. Si el agente cree que hacen falta, debe preguntar antes de agregarlas, no asumir.
7. **Corre en CPU, sin GPU dedicada.** Cualquier dependencia o configuración debe funcionar en el laptop de Josuram tal cual, sin asumir CUDA disponible. **(v2)** Si se elige un Raspberry Pi o la VM ARM de Oracle, también debe funcionar en Linux ARM64 (task 15.7).
8. **Verifica versiones antes de fijar una dependencia nueva.** Antes de agregar una librería a `requirements.txt`, confirma que soporta Python 3.14 y que es la última versión estable — no asumas que una versión recordada de entrenamiento sigue siendo la vigente.
9. **Presupuesto de hardware**: cualquier sugerencia de comprar hardware (cámaras, adaptadores, etc.) debe considerarse dentro de un presupuesto TOTAL de ~500.000 COP para todo el proyecto (compartido con los otros repos), no por componente. Si una sugerencia se acerca o supera ese total, dilo de inmediato, no después de construir toda la propuesta.
10. **Textos para humanos van en español**; nombres de variables, funciones, clases, commits y comentarios de código van en inglés, siguiendo el resto del proyecto.

## Herramientas que el agente puede usar libremente

- Lectura/escritura de archivos dentro de este repositorio.
- Ejecución de tests y scripts locales (`pytest`, scripts en `enrollment/`).
- Búsqueda web para verificar versiones de librerías o APIs de modelos (YOLO26n-pose, ArcFace/InsightFace, OSNet, SORT/DeepSORT).

## Herramientas que requieren confirmación explícita del usuario

- Cualquier llamada de red que no sea a `aforo-backend` o, en v2, al proyecto de Supabase del equipo (por ejemplo, subir datos a un servicio externo nuevo).
- Instalar dependencias del sistema operativo (fuera de `pip`).
- Modificar `config/pilot.yaml` con credenciales reales de la cámara WiFi o URLs de producción del backend.
- Exponer el servidor de video (sección 4.10) fuera de la red local (port forwarding, túneles tipo ngrok/Cloudflare, `0.0.0.0` en una red pública) o permitir acceso sin token.

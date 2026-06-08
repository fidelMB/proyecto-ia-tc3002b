# Compatibilidad Musical entre Usuarios de Spotify

## ¿Qué hace esto?

Este módulo extiende el proyecto principal para responder una pregunta orientada al usuario final:

> **¿Qué tan compatible es el gusto musical de dos personas, y qué canciones podrían disfrutar juntas?**

El análisis toma los modelos ya desarrollados en el proyecto (clustering K-Means, PCA, StandardScaler) y los aplica al perfil de audio de dos usuarios para calcular una **puntuación de compatibilidad** y generar **recomendaciones puente**.

---

## ¿Qué datos ofrece la API de Spotify?

Estos son los endpoints relevantes que permiten construir el perfil de audio de un usuario real:

| Endpoint | Datos obtenidos | Uso en el modelo |
|----------|----------------|------------------|
| `GET /me/top/tracks` | Top 50 canciones (corto / medio / largo plazo) | Construir el vector de perfil de audio |
| `GET /me/tracks` | Canciones guardadas (liked songs) | Enriquecer el perfil con más canciones |
| `GET /me/player/recently-played` | Últimas 50 canciones escuchadas | Capturar preferencias recientes |
| `GET /audio-features/{id}` | danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo | El vector numérico que alimenta el modelo |
| `GET /me/top/artists` | Artistas más escuchados + géneros | Inferir géneros preferidos implícitamente |
| `GET /recommendations` | Recomendaciones por seed (canciones / artistas / géneros) | Usar canciones puente como semilla |

> **Nota:** Spotify deprecó `GET /audio-features` para apps nuevas a partir de noviembre 2024. Para apps registradas antes de esa fecha el endpoint sigue activo. En este notebook se simula el perfil con el dataset existente del proyecto.

---

## Cómo funciona el análisis

```
Usuario A (50 canciones liked)          Usuario B (50 canciones liked)
         │                                        │
         ▼                                        ▼
  Vector de features de audio           Vector de features de audio
  (media de danceability, energy,       (media de danceability, energy,
   loudness, acousticness, etc.)         loudness, acousticness, etc.)
         │                                        │
         └──────────────┬─────────────────────────┘
                        ▼
              Similitud Coseno entre vectores
              → Puntuación de compatibilidad 0–100%
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
     Distribución de clusters   Punto medio de perfiles
     (K-Means k=2 del proyecto)  → Canciones más cercanas
                                   = Recomendaciones puente
```

### Métricas que produce el notebook

1. **Puntuación de compatibilidad** (0–100%) — similitud coseno normalizada entre perfiles de audio
2. **Distribución de clusters** — en qué proporción cada usuario escucha canciones enérgicas vs acústicas
3. **Radar chart** — comparación visual feature por feature
4. **Bar chart de divergencia** — qué dimensiones los separan más
5. **Top 10 canciones puente** — las canciones del dataset más cercanas al punto medio de ambos perfiles
6. **Visualización PCA** — posición de ambos usuarios y canciones puente en el espacio 2D

---

## Cómo correr el notebook

```bash
# Desde la raíz del proyecto
cd user_compatibility
jupyter notebook user_musical_compatibility.ipynb
```

Requiere las mismas dependencias del proyecto (`requirements.txt`).

---

## Cómo extender con datos reales de Spotify

```python
import spotipy
from spotipy.oauth2 import SpotifyOAuth

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="TU_CLIENT_ID",
    client_secret="TU_CLIENT_SECRET",
    redirect_uri="http://localhost:8888/callback",
    scope="user-top-read user-library-read user-read-recently-played"
))

def get_user_audio_profile(sp_client, limit=50, time_range="medium_term"):
    top = sp_client.current_user_top_tracks(limit=limit, time_range=time_range)
    ids = [t["id"] for t in top["items"]]
    features = sp_client.audio_features(ids)
    df = pd.DataFrame([f for f in features if f is not None])
    cols = ["danceability","energy","loudness","speechiness",
            "acousticness","instrumentalness","liveness","valence","tempo"]
    return df[cols].mean()

# Sustituir profile_a y profile_b en el notebook con:
# profile_a = get_user_audio_profile(sp_user_a)
# profile_b = get_user_audio_profile(sp_user_b)
```

---

## Modelos del proyecto reutilizados

| Notebook original | Técnica | Uso aquí |
|-------------------|---------|----------|
| `06_clustering.ipynb` | K-Means k=2 | Clasificar canciones de cada usuario en clusters |
| `04_pca.ipynb` | PCA 2D | Visualizar la posición de cada usuario en el espacio de audio |
| `02_cleaning_feature_engineering.ipynb` | StandardScaler | Escalar features antes de clustering y similitud |

---

## Ideas para extender el análisis

- **Playlist conjunta**: generar una playlist de N canciones balanceando los gustos de ambos usuarios
- **Compatibilidad dinámica**: recalcular el score con `recently_played` para ver si se acercan o alejan en el tiempo
- **Grupos de amigos**: extender a N usuarios y visualizar un heatmap de compatibilidad cruzada
- **Clasificador de popularidad**: usar el modelo de red neuronal del proyecto para predecir si las canciones puente serán populares para ambos usuarios

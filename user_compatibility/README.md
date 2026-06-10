# User Compatibility (extension)

An optional extension that reuses the main project's models (K-Means, PCA, StandardScaler)
to answer a user-facing question: **how compatible is the musical taste of two people, and
which songs could they enjoy together?**

`user_musical_compatibility.ipynb`:

- Builds an audio profile per user (the mean feature vector of their tracks).
- Scores compatibility with normalized cosine similarity between the two profiles.
- Shows which cluster each user listens to and which features separate them.
- Surfaces "bridge" tracks closest to the midpoint of both profiles.

User profiles are simulated from the dataset's genres. The notebook includes a commented
block showing how to replace the simulation with real profiles from the Spotify API
(`spotipy`). It depends on `data/processed/spotify_tracks_clean.csv` and the same
dependencies as the rest of the project.

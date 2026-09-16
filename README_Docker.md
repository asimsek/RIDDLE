# Publish the RIDDLE Docker image

Run these commands in the local terminal. Docker fetches the build files from the GitHub repo; no manual download, local repository clone, or Python virtual environment is needed.

> [!WARNING]
> These instructions document how to build and publish the runtime image for reference.<br>
> You do **NOT** need to rebuild the image to use this framework; use the pre-built image instead!

## 1. First-time setup

Install [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/), start it, and wait until the engine is running:

```bash
open -a Docker
docker info
```

Create a GitHub [personal access token (classic)](https://github.com/settings/tokens/new?scopes=write:packages) with `write:packages`.<br> 
At the password prompt, paste the token, not your GitHub password.<br>
Do not save the token in project files. See [GitHub authentication](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-with-a-personal-access-token-classic).

```bash
docker login ghcr.io -u asimsek
```

## 2. Build directly from GitHub and publish

Build from the files committed to `main`, not local edits. Docker fetches that revision internally.<br>
Wait for Docker Desktop to be ready before running the build. See [Docker remote builds](https://docs.docker.com/build/concepts/context/#git-repositories).

```bash
open -a Docker
docker buildx build \
  --platform linux/amd64 \
  --progress plain \
  --tag ghcr.io/asimsek/riddle-runtime:v1 \
  --push \
  'https://github.com/asimsek/RIDDLE.git#main'
```

Wait for the build and push to finish successfully before continuing.<br>
`--push` publishes the container package as `ghcr.io/asimsek/riddle-runtime:v1`; it does not modify the GitHub repository or create a GitHub Release. See [Docker build options](https://docs.docker.com/reference/cli/docker/buildx/build/).

Open the [package page](https://github.com/users/asimsek/packages/container/package/riddle-runtime).<br>
For NRP pulls without credentials, use **Package settings → Change visibility → Public**.<br>
New packages are private by default. See [GitHub package visibility](https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility#configuring-visibility-of-packages-for-your-personal-account).

## 3. Update an existing image

If the build files changed, commit and push them to GitHub separately, then repeat the build command in section 2.<br>
Keeping `v1` updates that package tag; no image deletion is needed.<br> 
Older image versions may remain in GitHub Packages.<br>
For a separate version, replace `v1` with `v2` in the build and digest commands.

## 4. Get the published image digest

After every successful publish or update, read registry metadata without downloading the image layers. See [Docker registry inspection](https://docs.docker.com/reference/cli/docker/buildx/imagetools/inspect/).

```bash
RIDDLE_DIGEST=$(docker buildx imagetools inspect ghcr.io/asimsek/riddle-runtime:v1 \
  --format '{{.Manifest.Digest}}') &&
RIDDLE_IMAGE="ghcr.io/asimsek/riddle-runtime@$RIDDLE_DIGEST"
echo "$RIDDLE_IMAGE"
```

Keep the printed `ghcr.io/asimsek/riddle-runtime@sha256:...` reference with your production records.<br>
Unlike a tag, a digest identifies an exact image. See [pulling by digest](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#pull-by-digest).

## 5. Use the updated image on NRP

Update `config/nrp/jupyter.yaml` file using the `RIDDLE_IMAGE` value above.<br>
For a fresh NRP setup, follow [README_NRP.md](README_NRP.md).

## 6. Cache and limited disk space

Keep cache when retrying: completed layers from a failed build can be reused. Avoid `--no-cache` when retrying so Docker can reuse completed build steps.<br>
Building from a GitHub URL still uses local Docker storage for source, image layers, and cache; it does not move the build to GitHub's servers. See [Docker build caching](https://docs.docker.com/build/cache/invalidation/).

Check usage:

```bash
df -h /System/Volumes/Data
docker system df
docker buildx du --builder desktop-linux
```

Optional, with no builds running: remove unused build cache last used more than 24 hours ago.<br> 
This can affect other projects' caches; removed layers must be rebuilt or downloaded again.<br>
It does not delete containers, volumes, or images stored in GitHub Container Registry (GHCR). See [Docker cache pruning](https://docs.docker.com/reference/cli/docker/buildx/prune/).

```bash
docker buildx prune --builder desktop-linux --filter "until=24h"
```

> [!CAUTION]
> Deep cleanup: `docker system prune -a --volumes` removes stopped containers, unused images and networks, build cache, and unused anonymous volumes.<br>
> Use only if the cleanup above has not freed enough space and you no longer need these resources. Deleted container and volume data cannot be recovered without a backup.





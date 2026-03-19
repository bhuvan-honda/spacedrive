# Create a SpaceDrive Docker
First, modifify the proxy used inside the docker in both `build.sh` and `deploy_local_docker.sh` if needed. 

In `deploy_local_docker.sh`, enter the dataset path and codebase path.

Then run:

```shell
# create docker image
bash ./build.sh

# deploy your created image
bash ./deploy_local_docker.sh
```



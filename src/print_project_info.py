import logging

import hydra
from encord import Project
from encord.client import EncordClientProject
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    project_client = EncordClientProject.initialise(
        cfg.project.project_hash, cfg.project.api_key
    )
    project = Project(project_client)

    print("# " * 40)
    print(f"# TITLE: {project.title}")
    print("# " * 40)
    print()

    print("# " * 40)
    print("# CLASSIFICATIONS")
    print("# " * 40)
    classifications = project.ontology["classifications"]
    indentation = ""
    for c in classifications:
        print(f"{indentation}{c['id']}: {c['name']} [{c['featureNodeHash']}]")

        indentation += "\t"

        for a in c["attributes"]:
            print(
                f"{indentation}{a['id']}: {a['name']} [{a['featureNodeHash']}]"
            )

            indentation += "\t"
            for o in a["options"]:
                print(
                    f"{indentation}{o['id']}: {o['label']} [{o['featureNodeHash']}]"
                )
            indentation = indentation[:-1]

        indentation = indentation[:-1]

    feature_node_hashes = []

    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
#SBATCH --job-name=L1C_scat
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/scratch/$USER/logs-sbatch/L1C-scat/L1C_scat_%j.out
#SBATCH --error=/scratch/$USER/logs-sbatch/L1C-scat/L1C_scat_%j.err
#SBATCH --partition=cpu
#SBATCH --mail-type=NONE

# =============================================================================
# Script: run_L1C_scat_coloc.slurm
# Description: Exécute do_IW_L1C_SAFE_from_L1B_SAFE avec colocalisation scatterometer
#              en utilisant une image Apptainer.
#
# Usage:
#   sbatch run_L1C_scat_coloc.slurm -s /path/to/SAFE -p B24 -o /output/dir \
#          -c /path/to/config.yml -i /path/to/image.sif [ -r ]
#
# Options:
#   -s | --safe           : Chemin vers le SAFE L1B XSP
#   -p | --productid      : ID du produit (ex: B24)
#   -o | --outputdir      : Répertoire de sortie
#   -c | --configprod     : Fichier de configuration YAML
#   -i | --apptainer-image : Chemin vers l'image Apptainer (.sif)
#   -r | --redo           : Réécrire les fichiers existants
#   -h | --help           : Affiche l'aide
# =============================================================================

# Créer le répertoire de logs
mkdir -p /scratch/$USER/logs-sbatch/L1C-scat

# -----------------------------------------------------------------------------
# Gestion des arguments
# -----------------------------------------------------------------------------
help() {
    cat << EOF
Usage: $0 -s SAFE -p PRODUCTID -o OUTPUTDIR -c CONFIGPROD -i APPTAINER_IMAGE [ -r ]

Options obligatoires:
  -s, --safe            Chemin vers le SAFE L1B XSP
  -p, --productid       ID du produit (ex: B24)
  -o, --outputdir       Répertoire de sortie
  -c, --configprod      Fichier de configuration YAML
  -i, --apptainer-image Chemin vers l'image Apptainer (.sif)

Options facultatives:
  -r, --redo            Réécrire les fichiers existants
  -h, --help            Affiche cette aide
EOF
    exit 0
}

# Parse avec getopt
SHORT="s:p:o:c:i:rh"
LONG="safe:,productid:,outputdir:,configprod:,apptainer-image:,redo,help"
OPTS=$(getopt -a -n "$0" --options $SHORT --longoptions $LONG -- "$@")
if [ $? != 0 ]; then
    echo "Erreur: échec du parsing des arguments"
    exit 1
fi
eval set -- "$OPTS"

while true; do
    case "$1" in
        -s | --safe )
            safe="$2"
            shift 2
            ;;
        -p | --productid )
            productid="$2"
            shift 2
            ;;
        -o | --outputdir )
            outputdir="$2"
            shift 2
            ;;
        -c | --configprod )
            configprod="$2"
            shift 2
            ;;
        -i | --apptainer-image )
            apptainer_image="$2"
            shift 2
            ;;
        -r | --redo )
            redo="--overwrite"
            shift
            ;;
        -h | --help )
            help
            ;;
        -- )
            shift
            break
            ;;
        * )
            echo "Option inconnue: $1"
            exit 1
            ;;
    esac
done

# Vérification des arguments obligatoires
if [ -z "$safe" ]; then
    echo "Erreur: --safe est obligatoire"
    help
fi
if [ -z "$productid" ]; then
    echo "Erreur: --productid est obligatoire"
    help
fi
if [ -z "$outputdir" ]; then
    echo "Erreur: --outputdir est obligatoire"
    help
fi
if [ -z "$configprod" ]; then
    echo "Erreur: --configprod est obligatoire"
    help
fi
if [ -z "$apptainer_image" ]; then
    echo "Erreur: --apptainer-image est obligatoire"
    help
fi

# Vérifier que l'image existe
if [ ! -f "$apptainer_image" ]; then
    echo "Erreur: Image Apptainer introuvable: $apptainer_image"
    exit 1
fi

# -----------------------------------------------------------------------------
# Affichage des paramètres
# -----------------------------------------------------------------------------
echo "======================================================================"
echo "Paramètres :"
echo "  SAFE          : $safe"
echo "  Product ID    : $productid"
echo "  Output dir    : $outputdir"
echo "  Config        : $configprod"
echo "  Apptainer image : $apptainer_image"
echo "  Overwrite     : ${redo:-Non}"
echo "======================================================================"

# -----------------------------------------------------------------------------
# Préparation du répertoire de sortie
# -----------------------------------------------------------------------------
mkdir -p "$outputdir"

# -----------------------------------------------------------------------------
# Construction et exécution de la commande
# -----------------------------------------------------------------------------
echo "Lancement avec Apptainer..."

# Bind mounts : adapter selon les chemins utilisés par vos données
BINDS="--bind /legacy/project/cersat/public:/legacy/project/cersat/public \
       --bind /scale/project/lops-siam-sentinel1-workbench:/scale/project/lops-siam-sentinel1-workbench \
       --bind /home/datawork-cersat-public:/home/datawork-cersat-public \
       --bind /scratch/$USER:/scratch/$USER"

CMD="apptainer exec $BINDS $apptainer_image \
     do_IW_L1C_SAFE_from_L1B_SAFE \
     --l1bsafe $safe \
     --outputdir $outputdir \
     --productid $productid \
     --configproducts $configprod \
     $redo"

echo "Commande: $CMD"
eval $CMD

# -----------------------------------------------------------------------------
# Vérification du code de retour
# -----------------------------------------------------------------------------
if [ $? -eq 0 ]; then
    echo "✅ Traitement terminé avec succès"
else
    echo "❌ Erreur lors du traitement"
    exit 1
fi
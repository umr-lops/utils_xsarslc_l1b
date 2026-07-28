import os
import logging
import glob
import numpy as np
import xarray as xr
from shapely.wkt import loads
from slcl1butils.get_config import get_conf
from slcl1butils.coloc.coloc_scatterometer import (find_matchup_file,resolve_scat_path,
                                                   read_scat_netcdf,get_scat_footprint_polygon,
                                                   associate_sar_tiles_with_scat,parse_tile_indices)
from slcl1butils.coloc.plot_scat_tile import plot_tile_with_scat_points
from slcl1butils.coloc.plot_scat_coloc import plot_colocation
logger = logging.getLogger('__name__')
# -----------------------------------------------------------------------------
# 4. Standalone test (main)
# -----------------------------------------------------------------------------
def entrypoint(config_path:str, product_id:str, no_scat_conf:bool, safename:str,
               outputdir_png:str, no_plot:bool=False, max_distance:float=50,
               plot_tile:str=None, radius_km:float=50, tile_output:str=None):
    """
    Main entry point for scatterometer colocation.
    
    Args:
        config_path (str): Path to YAML config file
        product_id (str): Product ID (e.g., B24)
        no_scat_conf (bool): Ignore scatterometer filters
        safename (str): Full path to L1B XSP SAFE
        outputdir_png (str): Directory to save PNG plots
        no_plot (bool): Skip plotting
        max_distance (float): Max distance in km for scat point
        plot_tile (str): Optional tile to plot individually (e.g., '0,3,1' or '3,1')
        radius_km (float): Radius in km for SCAT points around tile center
        tile_output (str): Output path for tile plot (PNG)
    """
    conf = get_conf(path_config_input=config_path)
    product_conf = conf.get(product_id)
    if not product_conf:
        logger.error("Product section not found.")
        exit(1)

    parquet_file = product_conf.get("scatterometer_matchups_file")
    scat_conf = product_conf.get("scatterometer", {})
    if not scat_conf.get("enabled", False):
        logger.info("Scatterometer colocation disabled.")
        exit(0)

    # Build filter
    if no_scat_conf:
        scat_filter = None
        logger.info("Scatterometer filters disabled.")
    else:
        scat_filter = {k: v for k, v in scat_conf.items() if k != "enabled"}
        logger.info("Using filters: %s", scat_filter)

    # Find matchup
    df_matchups = find_matchup_file(safename, parquet_file,
                                    scat_filter=scat_filter,
                                    time_window_seconds=3)
    if df_matchups is None or len(df_matchups) == 0:
        logger.error("No matchup found.")
        exit(1)

    row = df_matchups.iloc[0]
    scat_path = row['SCAT Path']

    # Resolve path with prefix replacement if needed
    scat_path = resolve_scat_path(scat_path)
    logger.info("Scatterometer file: %s", scat_path)

    # Load scatterometer data
    scat_ds = read_scat_netcdf(scat_path)

    # --- Récupérer le polygone SAR depuis le Parquet ---
    sar_footprint_wkt = row.get('SAR Footprint')
    sar_polygon = None
    if sar_footprint_wkt:
        
        try:
            sar_polygon = loads(sar_footprint_wkt)
            logger.info("SAR footprint polygon loaded from Parquet.")
        except Exception as e:
            logger.warning("Could not parse SAR Footprint WKT: %s", e)

    # --- Créer un polygone SCAT approximatif ---
    scat_polygon = get_scat_footprint_polygon(scat_ds, buffer_deg=0.5)

    # Find L1B files
    safe_dir = safename
    l1b_files = glob.glob(os.path.join(safe_dir, "*.nc"))
    if not l1b_files:
        logger.error("No L1B netCDF found in SAFE.")
        exit(1)
    l1b_files.sort()
    logger.info('Nb SAR XSP file found: %i',len(l1b_files))
    all_ds = []
    for l1b_path in l1b_files:
        logger.info("Loading SAR L1B from: %s =====================", l1b_path)
        try:
            ds_sar = xr.open_dataset(l1b_path, group='intraburst')
        except Exception as e:
            logger.error("Failed to open %s: %s", l1b_path, e)
            continue

        ds_sar = associate_sar_tiles_with_scat(ds_sar, scat_ds, max_distance_km=max_distance)

        matched_count = np.isfinite(ds_sar.scat_lon).sum().values
        total_tiles = np.prod(ds_sar.scat_lon.shape)
        valid_mask = np.isfinite(ds_sar.scat_wind_speed) & np.isfinite(ds_sar.scat_wind_direction)
        valid_count = valid_mask.sum().values
        logger.info("Subswath %s: Matched %d/%d tiles, valid wind %d/%d",
                    os.path.basename(l1b_path), matched_count, total_tiles, valid_count, total_tiles)

        all_ds.append((l1b_path, ds_sar))

        # --- Si une tuile spécifique est demandée, on la trace ---
        if plot_tile is not None:
            try:
                tile_idx = parse_tile_indices(plot_tile, ds_sar)
                logger.info("Plotting tile %s", tile_idx)
                
                # Déterminer le chemin de sortie
                if tile_output is None and outputdir_png:
                    base = os.path.basename(l1b_path).replace('.nc', f'_tile_{plot_tile.replace(",", "_")}.png')
                    tile_output = os.path.join(outputdir_png, base)
                    logger.info("Saving tile plot to %s", tile_output)
                
                plot_tile_with_scat_points(
                    ds_sar, scat_ds, tile_idx, 
                    radius_km=radius_km,
                    sar_polygon=sar_polygon,
                    scat_polygon=scat_polygon,
                    output_path=tile_output
                )
            except Exception as e:
                logger.error("Error plotting tile: %s", e)
            
            # Si on trace une tuile spécifique, on s'arrête après ce sous-swath
            # (on pourrait aussi continuer, mais c'est plus logique de s'arrêter)
            # if not no_plot:
            #     print("✅ Tile plot completed. Exiting.")
            #     return

        # --- Plot global du sous-swath ---
        if not no_plot:
            
            title = f"Scat colocation for {os.path.basename(l1b_path)}"
            
            output_path = None
            if outputdir_png:
                os.makedirs(outputdir_png, exist_ok=True)
                base = os.path.basename(l1b_path).replace('.nc', '_coloc.png')
                output_path = os.path.join(outputdir_png, base)
                logger.info("Saving plot to %s", output_path)
            
            plot_colocation(ds_sar, scat_ds, title=title,
                            sar_polygon=sar_polygon,
                            scat_polygon=scat_polygon,
                            output_path=output_path)
            
        
        logger.info('\n\n')

    print("✅ Colocation completed successfully.")


def main():
    import argparse
    import logging
    import matplotlib
    matplotlib.use('Agg')  # pour permettre la sauvegarde sans affichage

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Test scatterometer colocation")
    parser.add_argument("--config-path", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--product-id", type=str, required=True, help="Product ID")
    parser.add_argument("--safename", type=str, required=True, help="Full path to L1B XSP SAFE")
    parser.add_argument("--max-distance", type=float, default=50, help="Max distance (km)")
    parser.add_argument("--no-scat-conf", action="store_true", help="Ignore scatterometer filters")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--outputdir-png", type=str, default=None,
                        help="Directory to save PNG plots (optional). If not set, plots are displayed interactively.")
    
    # Nouveaux arguments pour le traçage d'une tuile individuelle
    parser.add_argument("--plot-tile", type=str, default=None,
                        help="Plot a specific tile: 'burst,tile_line,tile_sample' or 'tile_line,tile_sample'")
    parser.add_argument("--radius", type=float, default=50,
                        help="Radius in km for SCAT points around tile center (default: 50)")
    parser.add_argument("--tile-output", type=str, default=None,
                        help="Output path for tile plot (PNG). If not set, uses outputdir-png with auto-generated name.")
    
    args = parser.parse_args()
    
    entrypoint(config_path=args.config_path,
               product_id=args.product_id,
               no_scat_conf=args.no_scat_conf,
               safename=args.safename,
               outputdir_png=args.outputdir_png,
               no_plot=args.no_plot,
               max_distance=args.max_distance,
               plot_tile=args.plot_tile,
               radius_km=args.radius,
               tile_output=args.tile_output)


if __name__ == "__main__":
    main()
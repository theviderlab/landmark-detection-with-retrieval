import os
import urllib.request
import tarfile
import shutil
import pickle

def download_datasets(data_dir):
    """
    Descarga y prepara los datasets roxford5k y rparis6k si no existen.
    Elimina automáticamente las 20 imágenes corruptas de rparis6k.
    """
    import pickle

    os.makedirs(data_dir, exist_ok=True)
    datasets_dir = os.path.join(data_dir, 'datasets')
    os.makedirs(datasets_dir, exist_ok=True)

    datasets = ['roxford5k', 'rparis6k']
    base_urls = {
        'roxford5k': ('https://www.robots.ox.ac.uk/~vgg/data/oxbuildings', ['oxbuild_images-v1.tgz']),
        'rparis6k':  ('https://www.robots.ox.ac.uk/~vgg/data/parisbuildings', ['paris_1-v1.tgz', 'paris_2-v1.tgz'])
    }

    broken_rparis6k_files = {
        "paris_louvre_000136.jpg", "paris_louvre_000146.jpg", "paris_moulinrouge_000422.jpg",
        "paris_museedorsay_001059.jpg", "paris_notredame_000188.jpg", "paris_pantheon_000284.jpg",
        "paris_pantheon_000960.jpg", "paris_pantheon_000974.jpg", "paris_pompidou_000195.jpg",
        "paris_pompidou_000196.jpg", "paris_pompidou_000201.jpg", "paris_pompidou_000467.jpg",
        "paris_pompidou_000640.jpg", "paris_sacrecoeur_000299.jpg", "paris_sacrecoeur_000330.jpg",
        "paris_sacrecoeur_000353.jpg", "paris_triomphe_000662.jpg", "paris_triomphe_000833.jpg",
        "paris_triomphe_000863.jpg", "paris_triomphe_000867.jpg"
    }

    for dataset in datasets:
        if is_dataset_ready(dataset, data_dir):
            continue

        src_dir, dl_files = base_urls[dataset]
        dst_dir = os.path.join(datasets_dir, dataset, 'jpg')
        os.makedirs(dst_dir, exist_ok=True)

        for dl_file in dl_files:
            src_url = f"{src_dir}/{dl_file}"
            archive_path = os.path.join(dst_dir, dl_file)

            if not os.path.exists(archive_path.replace('.tgz', '')):  # evitar repetir si ya se extrajo
                print(f">> Descargando {dl_file} para {dataset}...")
                urllib.request.urlretrieve(src_url, archive_path)

                print(f">> Extrayendo {dl_file}...")
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tmp_dir = os.path.join(dst_dir, 'tmp')
                    os.makedirs(tmp_dir, exist_ok=True)
                    tar.extractall(path=tmp_dir)

                print(f">> Moviendo imágenes extraídas a {dst_dir}...")
                for root, _, files in os.walk(tmp_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            src_path = os.path.join(root, file)
                            dst_path = os.path.join(dst_dir, file)
                            shutil.move(src_path, dst_path)

                shutil.rmtree(tmp_dir)
                os.remove(archive_path)

        # Eliminar imágenes corruptas si el dataset es rparis6k
        if dataset == 'rparis6k':
            print(">> Eliminando imágenes corruptas conocidas de rparis6k...")
            for fname in broken_rparis6k_files:
                fpath = os.path.join(dst_dir, fname)
                if os.path.exists(fpath):
                    os.remove(fpath)
                    print(f"   - Eliminado: {fname}")

        # Descargar ground truth .pkl
        gnd_url = f"http://cmp.felk.cvut.cz/revisitop/data/datasets/{dataset}/gnd_{dataset}.pkl"
        gnd_path = os.path.join(datasets_dir, dataset, f"gnd_{dataset}.pkl")
        if not os.path.exists(gnd_path):
            print(f">> Descargando ground truth para {dataset}...")
            urllib.request.urlretrieve(gnd_url, gnd_path)

        # Filtrar imágenes corruptas del ground truth de rparis6k
        if dataset == 'rparis6k':
            print(">> Filtrando ground truth para remover imágenes corruptas...")
            with open(gnd_path, 'rb') as f:
                gnd = pickle.load(f)

            imlist = gnd['imlist']
            qimlist = gnd['qimlist']

            gnd_filtered = {
                'imlist': [im for im in imlist if im + ".jpg" not in broken_rparis6k_files],
                'qimlist': [qim for qim in qimlist if qim + ".jpg" not in broken_rparis6k_files],
                'gnd': [
                    g for i, g in enumerate(gnd['gnd'])
                    if qimlist[i] + ".jpg" not in broken_rparis6k_files
                ]
            }

            with open(gnd_path, 'wb') as f:
                pickle.dump(gnd_filtered, f)
            print(f">> Ground truth filtrado guardado en: {gnd_path}")

def download_distractors(data_dir):
    dataset = 'revisitop1m'
    nfiles = 100
    src_dir = 'http://ptak.felk.cvut.cz/revisitop/revisitop1m/jpg'
    dl_template = 'revisitop1m.{}.tar.gz'

    dataset_dir = os.path.join(data_dir, 'datasets', dataset)
    dst_dir = os.path.join(dataset_dir, 'jpg')
    tmp_dir = os.path.join(dataset_dir, 'jpg_tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    if not os.path.exists(dst_dir):
        print(f'>> Dataset {dataset} directory does not exist.\n>> Creating: {dst_dir}')
        for i in range(nfiles):
            dl_file = dl_template.format(i + 1)
            src_url = f'{src_dir}/{dl_file}'
            dst_file = os.path.join(tmp_dir, dl_file)

            if os.path.exists(dst_file.replace('.tar.gz', '')):
                print(f'>> [{i+1}/{nfiles}] Skipping {dl_file}, already extracted...')
                continue

            print(f'>> [{i+1}/{nfiles}] Downloading {dl_file}...')
            tmp_file = dst_file + '.tmp'
            while True:
                try:
                    urllib.request.urlretrieve(src_url, tmp_file)
                    os.rename(tmp_file, dst_file)
                    break
                except:
                    print('>>>> Download failed. Retrying...')

            print(f'>> [{i+1}/{nfiles}] Extracting {dl_file}...')
            with tarfile.open(dst_file, 'r:gz') as tar:
                tar.extractall(path=tmp_dir)
            os.remove(dst_file)

        os.rename(tmp_dir, dst_dir)

        # Descargar la lista de imágenes
        gnd_url = 'http://ptak.felk.cvut.cz/revisitop/revisitop1m/revisitop1m.txt'
        gnd_path = os.path.join(dataset_dir, 'revisitop1m.txt')
        if not os.path.exists(gnd_path):
            print(f'>> Downloading {dataset} image list...')
            urllib.request.urlretrieve(gnd_url, gnd_path)

def download_features(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    features_dir = os.path.join(data_dir, 'features')
    os.makedirs(features_dir, exist_ok=True)

    datasets = ['roxford5k', 'rparis6k']
    base_url = 'http://cmp.felk.cvut.cz/revisitop/data/features'

    for dataset in datasets:
        filename = f'{dataset}_resnet_rsfm120k_gem.mat'
        url = f'{base_url}/{filename}'
        dest_path = os.path.join(features_dir, filename)

        if not os.path.exists(dest_path):
            print(f'>> Downloading features for {dataset}: {filename}...')
            urllib.request.urlretrieve(url, dest_path)

def is_dataset_ready(dataset, data_dir):
    """
    Verifica si el dataset ya está completamente descargado y listo para usar.

    Args:
        dataset (str): Nombre del dataset (e.g., 'roxford5k', 'rparis6k').
        data_dir (str): Ruta absoluta a la carpeta raíz de los datos (e.g., TFM/assets/database)

    Returns:
        bool: True si está todo correcto, False si falta algo.
    """
    dataset_dir = os.path.join(data_dir, "datasets", dataset)
    jpg_dir = os.path.join(dataset_dir, "jpg")
    gnd_path = os.path.join(dataset_dir, f"gnd_{dataset}.pkl")

    if not os.path.exists(gnd_path):
        print(f">> Archivo ground truth no encontrado: {gnd_path}")
        return False

    if not os.path.isdir(jpg_dir):
        print(f">> Carpeta de imágenes no encontrada: {jpg_dir}")
        return False

    try:
        with open(gnd_path, 'rb') as f:
            cfg = pickle.load(f)
    except Exception as e:
        print(f">> Error al cargar ground truth: {e}")
        return False

    all_files = os.listdir(jpg_dir)
    all_set = set(all_files)

    # Revisar imágenes de base de datos y queries
    for name_list in [cfg.get("imlist", []), cfg.get("qimlist", [])]:
        for base_name in name_list:
            img_name = base_name + '.jpg'
            if img_name not in all_set:
                print(f">> Falta la imagen: {img_name}")
                return False

    return True
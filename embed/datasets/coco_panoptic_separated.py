import os.path as osp
import tempfile
from collections import OrderedDict

from embed import cv
import numpy as np
from embed.cv.utils import print_log
from terminaltables import AsciiTable
from PIL import Image
from tabulate import tabulate

from embed.datasets import DATASETS, CocoDataset

try:
    import pycocotools
    if not hasattr(pycocotools, '__sphinx_mock__'):  # for doc generation
        assert pycocotools.__version__ >= '12.0.2'
except AssertionError:
    raise AssertionError('Incompatible version of pycocotools is installed. '
                         'Run pip uninstall pycocotools first. Then run pip '
                         'install mmpycocotools to install open-mmlab forked '
                         'pycocotools.')

from functools import reduce

@DATASETS.register_module()
class CocoPanopticSeparatedDataset(CocoDataset):

    STUFF_CLASSES = ('thing', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff',
                     'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow',
                     'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
                     'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water',
                     'window-blind', 'window', 'tree', 'fence', 'ceiling', 'sky', 'cabinet', 'table', 'floor',
                     'pavement', 'mountain', 'grass', 'dirt', 'paper', 'food', 'building', 'rock', 'wall', 'rug')

    IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
           31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
           56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
           85, 86, 87, 88, 89, 90)

    STUFF_IDS = (0, 92, 93, 95, 100, 107, 109, 112, 118, 119, 122, 125, 128, 130, 133, 138, 141, 144, 145, 147, 148,
                 149, 151, 154, 155, 156, 159, 161, 166, 168, 171, 175, 176, 177, 178, 180, 181, 184, 185, 186, 187,
                 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200)

    def __init__(self, ann_file, seg_prefix=None, *args, **kwargs):
        super(CocoPanopticSeparatedDataset, self).__init__(ann_file, seg_prefix=seg_prefix, *args, **kwargs)

        self.pan_ann_file = ann_file.replace('instances', 'panoptic')
        assert seg_prefix is not None, seg_prefix
        self.pan_prefix = seg_prefix.replace('_stuff', '')
        if self.data_root is not None:
            if not osp.isabs(self.pan_ann_file):
                self.pan_ann_file = osp.join(self.data_root, self.pan_ann_file)
            if not osp.isabs(self.pan_prefix):
                self.pan_prefix = osp.join(self.data_root, self.pan_prefix)

        # Auto-generate panoptic coco stuff dataset
        if not osp.exists(self.seg_prefix):
            print(f'Auto-generating panoptic coco stuff dataset into {self.seg_prefix}...')

            from panopticapi.utils import rgb2id

            def process_panoptic_to_semantic(input_panoptic, output_semantic, segments, id_map):
                panoptic = np.asarray(Image.open(input_panoptic), dtype=np.uint32)
                panoptic = rgb2id(panoptic)
                output = np.zeros_like(panoptic, dtype=np.uint8) + 255
                for seg in segments:
                    cat_id = seg["category_id"]
                    new_cat_id = id_map[cat_id]
                    output[panoptic == seg["id"]] = new_cat_id
                Image.fromarray(output).save(output_semantic)

            def separate_coco_semantic_from_panoptic():
                '''
                id_map=dict()
                map(lambda id_th: id_map[id_th] = 0, self.IDS)
                '''
                id_map = {id_th: 0 for id_th in self.IDS}
                id_map.update(self.stuff_ids2label)
                id_map[0] = 255
                obj = cv.load(self.pan_ann_file)
                pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))
                def iter_annotations():
                    for anno in obj["annotations"]:
                        file_name = anno["file_name"]
                        segments = anno["segments_info"]
                        input = os.path.join(self.pan_prefix, file_name)
                        output = os.path.join(self.seg_prefix, file_name)
                        yield input, output, segments
                print("Start writing to {} ...".format(sem_seg_root))
                start = time.time()
                pool.starmap(
                    functools.partial(process_panoptic_to_semantic, id_map=id_map),
                    iter_annotations(),
                    chunksize=100,
                )
                print("Finished. time: {:.2f}s".format(time.time() - start))

            os.makedirs(self.seg_prefix)
            self.separate_coco_semantic_from_panoptic()

    def load_annotations(self, ann_file):
        self.STUFF_CLASSES = CocoPanopticSeparatedDataset.STUFF_CLASSES
        self.num_stuff_cls = len(self.STUFF_CLASSES)
        assert self.num_stuff_cls == 54, (self.num_stuff_cls, 54)

        self.IDS = CocoPanopticSeparatedDataset.IDS
        assert len(self.IDS) == 80, (self.IDS, 80)
        self.STUFF_IDS = CocoPanopticSeparatedDataset.STUFF_IDS
        assert len(self.STUFF_IDS) == self.num_stuff_cls, (len(self.STUFF_IDS), self.num_stuff_cls)
        self.stuff_ids2label = {cat_id: i for i, cat_id in enumerate(self.STUFF_IDS)}
        self.ignore_label = 255

        return super(CocoPanopticSeparatedDataset, self).load_annotations(ann_file)

    def format_results(self, results, jsonfile_prefix=None, key='instance', **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            key (str): The evaluation type. instance, sem_seg or panoptic.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix, key)
        return result_files, tmp_dir

    def results2json_instance(self, results, outfile_prefix):
        result_files = dict(bbox=f'{outfile_prefix}.bbox.json',
                            segm=f'{outfile_prefix}.segm.json')
        json_results = list(map(lambda idx, result:
                                   dict(image_id=self.img_ids[idx],
                                        bbox=self.xyxy2xywh(result['pred_bbox']),
                                        score=float(result['score']),
                                        category_id=self.cat_ids[result['cls']]),
                                *enumerate(results)))
        cv.dump(json_results, result_files['bbox'])
        '''
        map(lambda json_result, result: json_result['segmentation'] = result['pred_mask'],
            *zip(json_results, results))
        '''
        for json_result, result in zip(json_results, results):
            json_result['segmentation'] = result['pred_mask']
        cv.dump(json_results, result_files['segm'])
        return result_files

    def results2json_stuff(self, results, outfile_prefix):
        result_files = dict(sem_seg=f'{outfile_prefix}.sem_seg.json')
        import pycocotools.mask as mask_util
        def encode_json_stuff(result, data_info):
            def encode_json_stuff_per_cls(cls, sem_seg, input_file_name):
                dataset_id = self.cat_ids_stuff[cls]
                mask = (sem_seg == label).astype(np.uint8)
                mask_rle = mask_util.encode(np.array(mask[:, :, np.newaxis], order="F"))[0]
                mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
                return dict(category_id=dataset_id, segmentation=mask_rle, file_name=input_file_name)
            # TODO(ljm): maybe list(map(partial())) here
            return multi_apply(encode_json_stuff_per_cls, result['cls'],
                                   sem_seg=result['pred_sem'], input_file_name=data_info['filename'])
        json_results = list(map(encode_json_stuff, results, self.data_infos))
        json_results = reduce(lambda a, b: a + b, json_results)
        cv.dump(json_results, result_files['sem_seg'])
        return result_files

    def results2json_panoptic(self, results, outfile_prefix):
        from panopticapi.utils import id2rgb

        result_files = dict(panoptic=f'{outfile_prefix}.panoptic.json')

        def process_per_image(idx, result):
            panoptic_seg, segments_info = result['panoptic_seg'], result['segments_info']
            panoptic_png = self.data_infos[idx]['filename'].replace('jpg', 'png')
            outfile_path = f'{outfile_prefix}/panoptic/{panoptic_png}'
            Image.fromarray(id2rgb(panoptic_seg)).save(outfile_path)

            def convert_cat_id(segment_info):
                isthing, cont_id = segment_info.pop('isthing'), segment_info['category_id']
                segment_info['category_id'] = self.cat_ids[cont_id] if isthing else \
                                              self.STUFF_IDS[cont_id]

            list(map(convert_cat_id, segments_info))
            return dict(image_id=self.img_ids[idx],
                        file_name=panoptic_png,
                        segments_info=segments_info)

        prediction_results = reduce(lambda a, b: a + b,
                                    map(process_per_image, *enumerate(results)))

        json_results = cv.load(self.pan_ann_file)
        json_results['annotations'] = prediction_results
        cv.dump(json_results, result_files['panoptic'])
        return result_files

    def results2json(self, results, jsonfile_prefix, key='instance'):
        results = list(map(lambda result: result.pop(key), results))
        result_files = eval('self.results2json_{key}')(results, jsonfile_prefix)
        return result_files

    def evaluate(self,
                 results,
                 metric='bbox',
                 **kwargs):
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast', 'miou', 'panoptic']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        instance_metrics = filter(lambda metric: metric not in ['miou', 'panoptic'], metrics)
        extra_metrics = filter(lambda metric: metric in ['miou', 'panoptic'], metrics)

        eval_results = self.evaluate_instance(results, instance_metrics, **kwargs)

        # TODO(ljm): **kwargs OK here?
        multi_apply(lambda extra_metric, results, **kwargs:
                        eval_results.update(
                            eval('self.evaluate_{extra_metric}')(results, **kwargs)),
                    extra_metrics, results=results, **kwargs)

        return eval_results

    def evaluate_stuff(self, results,
                               logger=None,
                               jsonfile_prefix=None,
                               **kwargs):
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix, key='miou')
        msg = str('\n' if logger is None else '') + f'Evaluating miou...'
        print_log(msg, logger=logger)

        annos_stuff = list(map(lambda img_info: img_info['filename'].replace('jpg', 'png'),
                               self.data_infos))

        def get_conf_mat_single_image(result, sem_seg_file, minlength=54**2):
            gt = np.asarray(Image.open(sem_seg_file), dtype=np.int64)
            gt[gt == self.ignore_label] = self._num_classes
            return np.bincount(
                       (self.num_stuff_cls + 1) * result['pred_mask'].reshape(-1) + gt.reshape(-1),
                       minlength=minlength,
                   )

        '''
        conf_mat = reduce(lambda a, b: a + b,
                          multi_apply(get_conf_mat_single_image, *zip(results, annos_stuff),
                                                                 minlength=(1 + self.num_stuff_cls)**2)
                   .reshape(1 + self.num_stuff_cls, 1 + self.num_stuff_cls)
        '''

        conf_mat = reduce(lambda a, b: a + b,
                          map(partial(get_conf_mat_single_image,
                                      minlength=(1 + self.num_stuff_cls)**2),
                              results, annos_stuff)) \
                   .reshape(1 + self.num_stuff_cls, 1 + self.num_stuff_cls)

        acc = np.full(self._num_cls_stuff, np.nan, dtype=np.float)
        iou = np.full(self._num_cls_stuff, np.nan, dtype=np.float)
        tp = conf_mat.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(conf_mat[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(conf_mat[:-1, :-1], axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        eval_results = OrderedDict()
        eval_results["mIoU"] = 100 * miou
        eval_results["fwIoU"] = 100 * fiou
        for i, name in enumerate(self.STUFF_CLASSES):
            eval_results["IoU-{}".format(name)] = 100 * iou[i]
        eval_results["mACC"] = 100 * macc
        eval_results["pACC"] = 100 * pacc
        for i, name in enumerate(self.STUFF_CLASSES):
            eval_results["ACC-{}".format(name)] = 100 * acc[i]

        def print_stuff_results(eval_results, logger):
            # TODO(ljm): remove the first column
            headers = ["", "IoU", "ACC", "#categories"]
            data = []
            for name in self.STUFF_CLASSES:
                row = [name,  eval_results[f"IoU-{name}"], eval_results[f"ACC-{name}"], name]
                data.append(row)
            data.append("mean", eval_results["mIoU"], eval_results["mACC"], "mean")
            data.append("weighted iou/tp acc", eval_results["fwIoU"], eval_results["pACC"], "weighted iou/tp acc")
            from tabulate import tabulate
            table = tabulate(
                data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
            )
            print_log("\nSegmentation Results:\n" + table, logger=logger)

        print_stuff_results(eval_results, logger)

        return OrderedDict(sem_seg=eval_results)

    def evaluate_panoptic(self, results,
                                logger=None,
                                jsonfile_prefix=None,
                                **kwargs):
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix, key='panoptic')
        msg = str('\n' if logger is None else '') + f'Evaluating panoptic...'
        print_log(msg, logger=logger)

        from panopticapi.evaluation import pq_compute

        # TODO(ljm): see what will it output
        # with contextlib.redirect_stdout(io.StringIO()):
        pq_res = pq_compute(
            self.pan_ann_file,
            result_files['panoptic'],
            gt_folder=self.pan_prefix,
            pred_folder=result_files['panoptic'].replace('.json', ''),
        )

        eval_results = OrderedDict()
        eval_results["PQ"] = 100 * pq_res["All"]["pq"]
        eval_results["SQ"] = 100 * pq_res["All"]["sq"]
        eval_results["RQ"] = 100 * pq_res["All"]["rq"]
        eval_results["PQ_th"] = 100 * pq_res["Things"]["pq"]
        eval_results["SQ_th"] = 100 * pq_res["Things"]["sq"]
        eval_results["RQ_th"] = 100 * pq_res["Things"]["rq"]
        eval_results["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        eval_results["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        eval_results["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

        def print_panoptic_results(pq_res, logger):
            headers = ["", "PQ", "SQ", "RQ", "#categories"]
            data = []
            for name in ["All", "Things", "Stuff"]:
                row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
                data.append(row)
            from tabulate import tabulate
            table = tabulate(
                data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
            )
            print_log("\nPanoptic Evaluation Results:\n" + table, logger=logger)

        print_panoptic_results(pq_res, logger)

        return OrderedDict(panoptic=eval_results)

    def evaluate_instance(self,
                          results,
                          metric='bbox',
                          logger=None,
                          jsonfile_prefix=None,
                          classwise=False,
                          proposal_nums=(100, 300, 1000),
                          iou_thrs=None,
                          metric_items=None,
                          **kwargs):
        eval_results = super(CocoPanopticSeparatedDataset, self).evaluate(results,
                                                                          basic_metrics,
                                                                          logger,
                                                                          jsonfile_prefix,
                                                                          classwise,
                                                                          proposal_nums,
                                                                          iou_thrs,
                                                                          metric_items)
        return OrderedDict(instance=eval_results)

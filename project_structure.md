卷 F盘 的文件夹 PATH 列表
卷序列号为 B666-F270
F:.
|   .gitignore
|   coco的数据集下载地址.md
|   decision_maker.py
|   Dockerfile.py
|   exp-s-4358.pt
|   export_custom_model_to_onnx.py
|   Filter.py
|   LICENSE
|   main_simulation_loop.py
|   my_pybullet_test.py
|   obstacle_detector.py
|   pybullet_sim.py
|   pybullet_test.py
|   README.md
|   requirements.txt
|   test01.py
|   test_model_build.py
|   yolov5s.pt
|   yolov5s_baseline_structure.onnx
|   yolov5s_with_se_structure.onnx
|   yolov5_embodied_ai_architecture
|   具体过程.md
|
+---.idea
|   |   .gitignore
|   |   CNN_project_AI_course.iml
|   |   misc.xml
|   |   modules.xml
|   |   vcs.xml
|   |   workspace.xml
|   |
|   \---inspectionProfiles
|           profiles_settings.xml
|
+---Code_from_CHI_Xu
|       attention_blocks.py
|       common_v2_SEBlock_and_CBAM.py
|       common_version1_SEBlock.py
|       decision_maker.py
|       diagram.py
|       generate_project_graph.py
|       mock_detector.py
|       project_arch_diagrams_simple_labels
|       yolov5s_CBAM_only_neck.yaml
|       yolov5s_neck_only_se.yaml
|       yolov5s_se_backbone_and_neck.yaml
|       yolov5s_se_backbone_r32.yaml
|       yolov5s_se_backbone_r8.yaml
|       yolov5s_with_se.yaml
|       yolov5_embodied_ai_architecture
|       yolov5_project_flow_diagrams
|       yolov5_project_flow_diagrams.png
|       yolo_version1_SEBlocks.py
|       yolo_version2_CBAM_and_SEBlock.py
|
+---daliy_work_records
|       Chi_Xu.md
|
+---labels
|   +---train2014
|   |       classes.txt
|   |
|   \---val2014
|           classes.txt
|
+---yolov5
|   |   .dockerignore
|   |   .gitattributes
|   |   .gitignore
|   |   benchmarks.py
|   |   CITATION.cff
|   |   CONTRIBUTING.md
|   |   detect.py
|   |   export.py
|   |   hubconf.py
|   |   LICENSE
|   |   pyproject.toml
|   |   README.md
|   |   README.zh-CN.md
|   |   requirements.txt
|   |   train.py
|   |   tutorial.ipynb
|   |   val.py
|   |
|   +---.github
|   |   |   dependabot.yml
|   |   |
|   |   +---ISSUE_TEMPLATE
|   |   |       bug-report.yml
|   |   |       config.yml
|   |   |       feature-request.yml
|   |   |       question.yml
|   |   |
|   |   \---workflows
|   |           ci-testing.yml
|   |           cla.yml
|   |           docker.yml
|   |           format.yml
|   |           links.yml
|   |           merge-main-into-prs.yml
|   |           stale.yml
|   |
|   +---classify
|   |       predict.py
|   |       train.py
|   |       tutorial.ipynb
|   |       val.py
|   |
|   +---data
|   |   |   Argoverse.yaml
|   |   |   coco.yaml
|   |   |   coco128-seg.yaml
|   |   |   coco128.yaml
|   |   |   GlobalWheat2020.yaml
|   |   |   ImageNet.yaml
|   |   |   ImageNet10.yaml
|   |   |   ImageNet100.yaml
|   |   |   ImageNet1000.yaml
|   |   |   Objects365.yaml
|   |   |   SKU-110K.yaml
|   |   |   VisDrone.yaml
|   |   |   VOC.yaml
|   |   |   xView.yaml
|   |   |
|   |   +---hyps
|   |   |       hyp.no-augmentation.yaml
|   |   |       hyp.Objects365.yaml
|   |   |       hyp.scratch-high.yaml
|   |   |       hyp.scratch-low.yaml
|   |   |       hyp.scratch-med.yaml
|   |   |       hyp.VOC.yaml
|   |   |
|   |   +---images
|   |   |       bus.jpg
|   |   |       zidane.jpg
|   |   |
|   |   \---scripts
|   |           download_weights.sh
|   |           get_coco.sh
|   |           get_coco128.sh
|   |           get_imagenet.sh
|   |           get_imagenet10.sh
|   |           get_imagenet100.sh
|   |           get_imagenet1000.sh
|   |
|   +---models
|   |   |   common.py
|   |   |   experimental.py
|   |   |   our_modules.py
|   |   |   tf.py
|   |   |   yolo.py
|   |   |   yolov5l.yaml
|   |   |   yolov5m.yaml
|   |   |   yolov5n.yaml
|   |   |   yolov5s.yaml
|   |   |   yolov5x.yaml
|   |   |   __init__.py
|   |   |
|   |   +---hub
|   |   |       anchors.yaml
|   |   |       yolov3-spp.yaml
|   |   |       yolov3-tiny.yaml
|   |   |       yolov3.yaml
|   |   |       yolov5-bifpn.yaml
|   |   |       yolov5-fpn.yaml
|   |   |       yolov5-p2.yaml
|   |   |       yolov5-p34.yaml
|   |   |       yolov5-p6.yaml
|   |   |       yolov5-p7.yaml
|   |   |       yolov5-panet.yaml
|   |   |       yolov5l6.yaml
|   |   |       yolov5m6.yaml
|   |   |       yolov5n6.yaml
|   |   |       yolov5s-ghost.yaml
|   |   |       yolov5s-LeakyReLU.yaml
|   |   |       yolov5s-transformer.yaml
|   |   |       yolov5s6.yaml
|   |   |       yolov5x6.yaml
|   |   |
|   |   +---segment
|   |   |       yolov5l-seg.yaml
|   |   |       yolov5m-seg.yaml
|   |   |       yolov5n-seg.yaml
|   |   |       yolov5s-seg.yaml
|   |   |       yolov5x-seg.yaml
|   |   |
|   |   \---__pycache__
|   |           common.cpython-312.pyc
|   |           experimental.cpython-312.pyc
|   |           yolo.cpython-312.pyc
|   |           __init__.cpython-312.pyc
|   |
|   +---segment
|   |       predict.py
|   |       train.py
|   |       tutorial.ipynb
|   |       val.py
|   |
|   \---utils
|       |   activations.py
|       |   augmentations.py
|       |   autoanchor.py
|       |   autobatch.py
|       |   callbacks.py
|       |   dataloaders.py
|       |   downloads.py
|       |   general.py
|       |   loss.py
|       |   metrics.py
|       |   plots.py
|       |   torch_utils.py
|       |   triton.py
|       |   __init__.py
|       |
|       +---aws
|       |       mime.sh
|       |       resume.py
|       |       userdata.sh
|       |       __init__.py
|       |
|       +---docker
|       |       Dockerfile
|       |       Dockerfile-arm64
|       |       Dockerfile-cpu
|       |
|       +---flask_rest_api
|       |       example_request.py
|       |       README.md
|       |       restapi.py
|       |
|       +---google_app_engine
|       |       additional_requirements.txt
|       |       app.yaml
|       |       Dockerfile
|       |
|       +---loggers
|       |   |   __init__.py
|       |   |
|       |   +---clearml
|       |   |       clearml_utils.py
|       |   |       hpo.py
|       |   |       README.md
|       |   |       __init__.py
|       |   |
|       |   +---comet
|       |   |       comet_utils.py
|       |   |       hpo.py
|       |   |       optimizer_config.json
|       |   |       README.md
|       |   |       __init__.py
|       |   |
|       |   \---wandb
|       |           wandb_utils.py
|       |           __init__.py
|       |
|       +---segment
|       |       augmentations.py
|       |       dataloaders.py
|       |       general.py
|       |       loss.py
|       |       metrics.py
|       |       plots.py
|       |       __init__.py
|       |
|       \---__pycache__
|               augmentations.cpython-312.pyc
|               autoanchor.cpython-312.pyc
|               dataloaders.cpython-312.pyc
|               downloads.cpython-312.pyc
|               general.cpython-312.pyc
|               metrics.cpython-312.pyc
|               plots.cpython-312.pyc
|               torch_utils.cpython-312.pyc
|               __init__.cpython-312.pyc
|
+---__pycache__
|       decision_maker.cpython-312.pyc
|       obstacle_detector.cpython-312.pyc
|       pybullet_sim.cpython-312.pyc
|
+---基线模型相关
|       coco2yolo_and_filter.py
|       说明.md
|
\---筛选后数据集相关
        coco2yolo.py
        Filter.py
        具体过程.md


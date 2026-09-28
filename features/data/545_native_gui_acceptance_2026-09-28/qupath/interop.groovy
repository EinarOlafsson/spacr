import com.google.gson.GsonBuilder
import javafx.application.Platform
import javafx.animation.PauseTransition
import javafx.util.Duration
import javafx.embed.swing.SwingFXUtils
import javax.imageio.ImageIO
import qupath.lib.gui.QuPathGUI
import qupath.lib.gui.prefs.PathPrefs
import qupath.lib.gui.commands.InteractiveObjectImporter
import qupath.lib.io.PathIO

final root = new File('/mnt/wd4tb/scratch/spacr-completion/545-native-interop-preparation')
final output = new File(root, 'qupath')
final gson = new GsonBuilder().setPrettyPrinting().create()
final failure = { Throwable error ->
    new File(output, 'failure.txt').withPrintWriter { writer -> error.printStackTrace(writer) }
    System.err.println('QUPATH_INTEROP_FAILED: ' + error)
    System.exit(1)
}
try {
    assert Platform.isFxApplicationThread(): 'Native startup script must run on the GUI thread'
    def qupath = QuPathGUI.getInstance()
    assert qupath != null && qupath.getStage().isShowing()
    qupath.getStage().setWidth(1280)
    qupath.getStage().setHeight(900)
    qupath.getStage().setX(0)
    qupath.getStage().setY(0)
    PathPrefs.imageTypeSettingProperty().set(PathPrefs.ImageTypeSetting.AUTO_ESTIMATE)
    assert qupath.openImage(qupath.getViewer(), new File(root, 'field.tif').absolutePath, false, false)
    def data = qupath.getImageData()
    assert data.getServer().getWidth() == 96 && data.getServer().getHeight() == 64
    assert InteractiveObjectImporter.promptToImportObjectsFromFile(data, new File(root, 'spacr-field.geojson'))
    def objects = data.getHierarchy().getAnnotationObjects().toList()
    def fixture = gson.fromJson(new File(root, 'fixture-receipt.json').text, Map)
    def source = gson.fromJson(new File(root, 'spacr-field.geojson').text, Map)
    assert objects.size() == fixture.expected_roi_count
    assert data.getHierarchy().getDetectionObjects().isEmpty()
    def labels = ['cell', 'nucleus'].collectEntries { name ->
        [(name): ImageIO.read(new File(root, "expected-${name}.tif"))]
    }
    def observed = fixture.objects.collect { expected ->
        expected.object_id = expected.object_id.intValue()
        def name = "${expected.object_type} ${expected.object_id}".toString()
        def matches = objects.findAll { it.getName() == name }
        assert matches.size() == 1: "Missing/duplicated object ${name}"
        def object = matches[0]
        def roi = object.getROI()
        assert object.getPathClass().toString() == expected['class']
        assert object.getMeasurementList().get('object_id') == expected.object_id
        assert roi.getGeometry().isValid()
        def bounds = [roi.getBoundsX(), roi.getBoundsY(), roi.getBoundsWidth(), roi.getBoundsHeight()]
        assert bounds == expected.bounds_xywh: "Bounds differ for ${name}: ${bounds}"
        assert Math.abs(roi.getArea() - expected.area_pixels) < 1e-8
        def sourceFeature = source.get('features').find { feature -> feature.get('properties').get('name').toString().equals(name) }
        assert sourceFeature != null: "No source feature for ${name}; names=${source.get('features').collect { feature -> feature.get('properties').get('name') }}"
        assert object.getID().toString() == sourceFeature.get('id')
        def raster = labels[expected.object_type].getRaster()
        int differences = 0
        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 96; x++) {
                boolean wanted = raster.getSample(x, y, 0) == expected.object_id
                if (roi.contains(x + 0.5, y + 0.5) != wanted) differences++
            }
        }
        assert differences == 0: "Geometry differs at ${differences} pixel centres for ${name}"
        [name:name, object_id:expected.object_id, classification:object.getPathClass().toString(),
         bounds_xywh:bounds, area_pixels:roi.getArea(), pixel_centre_differences:differences,
         geometry_type:roi.getGeometry().getGeometryType(), uuid:object.getID().toString()]
    }
    def annotationsTab = qupath.getAnalysisTabPane().getTabs().find { it.getText() == 'Annotations' }
    assert annotationsTab != null
    qupath.getAnalysisTabPane().getSelectionModel().select(annotationsTab)
    qupath.getViewer().setDownsampleFactor(0.15)
    qupath.getViewer().setCenterPixelLocation(48, 32)
    PathIO.writeImageData(new File(output, 'imported.qpdata'), data)
    def delay = new PauseTransition(Duration.seconds(3))
    delay.setOnFinished {
        try {
            assert qupath.getStage().isShowing()
            assert qupath.getImageData().getHierarchy().getAnnotationObjects().size() == 8
            def screenshot = qupath.getStage().getScene().snapshot(null)
            assert ImageIO.write(SwingFXUtils.fromFXImage(screenshot, null), 'png', new File(output, 'qupath-gui.png'))
            def receipt = [schema:'spacr.545.qupath_gui_interop.v1', passed:true,
                qupath_version:'0.7.0', source_revision:new File(root, 'source-revision.txt').text.trim(),
                actual_gui_visible:qupath.getStage().isShowing(), javafx_application_thread:Platform.isFxApplicationThread(),
                native_import_api:'InteractiveObjectImporter.promptToImportObjectsFromFile',
                input:'spacr-field.geojson', image_size_xy:[96,64], annotation_count:objects.size(),
                objects:observed, screenshot:'qupath-gui.png', saved_native_data:'imported.qpdata',
                synthetic_fixture:true, rendering:'JavaFX prism software; private Xvfb',
                archive_sha256:'165e27a0731d58ba039e9d0d34a54acf896bb92e81922370df295cb85418ce53']
            new File(output, 'gui-receipt.json').text = gson.toJson(receipt) + '\n'
            new File(output, 'GUI_READY').text = 'Verified native import and geometry; GUI remains visible for external screenshot.\n'
            def exitDelay = new PauseTransition(Duration.seconds(12))
            exitDelay.setOnFinished { System.exit(0) }
            exitDelay.play()
        } catch (Throwable error) { failure(error) }
    }
    delay.play()
} catch (Throwable error) { failure(error) }

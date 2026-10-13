// Actual GUI interoperability: batch mode is explicitly off.
out = "/mnt/wd4tb/scratch/spacr-completion/545-native-interop-preparation/";
setBatchMode(false);
open(out + "field.tif");
rename("spaCR 545 source image - real ImageJ GUI");
roiManager("reset");
roiManager("open", out + "spacr-field.zip");
count = roiManager("count");
csv = "name,x,y,width,height,area,selection_type,class,object_id,object_type\n";
for (i = 0; i < count; i++) {
    roiManager("select", i);
    name = Roi.getName;
    getSelectionBounds(x, y, w, h);
    getStatistics(area);
    csv += name + "," + x + "," + y + "," + w + "," + h + "," + area + "," + selectionType + "," + Roi.getProperty("class") + "," + Roi.getProperty("object_id") + "," + Roi.getProperty("object_type") + "\n";
}
File.saveString(csv, out + "imagej-native-objects.csv");
roiManager("deselect");
roiManager("save", out + "imagej-resaved-roiset.zip");
types = newArray("cell", "nucleus");
for (t = 0; t < types.length; t++) {
    kind = types[t];
    newImage("ImageJ rasterized " + kind, "16-bit black", 96, 64, 1);
    for (i = 0; i < count; i++) {
        roiManager("select", i);
        name = Roi.getName;
        if (startsWith(name, kind + "-")) {
            parts = split(name, "-");
            setColor(parseInt(parts[1]));
            fill();
        }
    }
    run("Select None");
    saveAs("Tiff", out + "imagej-rasterized-" + kind + ".tif");
}
selectWindow("spaCR 545 source image - real ImageJ GUI");
roiManager("deselect");
roiManager("show all with labels");
for (i = 0; i < 5; i++) run("In [+]");
File.saveString("version=" + getVersion() + "\ncount=" + count + "\nbatch_mode=false\n", out + "imagej-native-status.txt");
// Keep real windows alive for the external Xvfb screenshot, then exit normally.
for (i = 0; i < 300; i++) {
    if (File.exists(out + "screenshot-complete.txt")) break;
    wait(100);
}
run("Quit");

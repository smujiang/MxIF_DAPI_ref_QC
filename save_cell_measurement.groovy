import qupath.lib.gui.QuPathGUI
import qupath.lib.scripting.QP
import qupath.lib.gui.tools.MeasurementExporter
import qupath.lib.objects.PathCellObject


//Use either "project" OR "outputFolder" to determine where your detection files will go
// def project = QuPathGUI.getInstance().getProject().getBaseDirectory()

output_dir = "\\\\mfad\\researchmn\\HCPR\\HCPR-GYNECOLOGICALTUMORMICROENVIRONMENT\\Multiplex_Img\\OVCA_TMA22\\cell_mesurements"

def imageData = QP.getCurrentImageData()
def hierarchy = imageData.getHierarchy()
def server = imageData.getServer()
//
// def annotations = getAnnotationObjects()
// int i = 1
// for (annotation in annotations)
// {
//     if(annotation.getPathClass().toString().equals("ROI")){
//         int cx = (int)annotation.getROI().x
//         int cy = (int)annotation.getROI().y
//         hierarchy.getSelectionModel().clearSelection()
//         selectObjects{p -> p == annotation}
//         F = new File(server.getMetadata().getPath())
//         print(F.getName().lastIndexOf('.'))
//         pathOutput = output_dir + File.separator + F.getName().take(F.getName().lastIndexOf('.'))
//         File newDir = new File(pathOutput)
//         if (!newDir.exists()) {
//             newDir.mkdirs()
//         }
//         save_fn = pathOutput + File.separator + String.format( "%d", cx) + "_" + String.format( "%d", cy)  + "_detections.txt"
//         print(save_fn)
//         saveDetectionMeasurements(save_fn)
//     }
// }


// Get the list of all images in the current project
def project = getProject()
def imagesToExport = project.getImageList()

// Separate each measurement value in the output file with a tab ("\t")
def separator = "\t"

// Choose the columns that will be included in the export
// Note: if 'columnsToInclude' is empty, all columns will be included
// def columnsToInclude = new String[]{"Name", "Class", "Nucleus: Area"}

// Choose the type of objects that the export will process
// Other possibilities include:
//    1. PathAnnotationObject
//    2. PathDetectionObject
//    3. PathRootObject
// Note: import statements should then be modified accordingly
def exportType = PathCellObject.class
// def exportType = PathDetectionObject.class

// Choose your *full* output path
File F = new File(server.getMetadata().getName())
def pathOutput = output_dir + File.separator + F.getName().take(F.getName().lastIndexOf('.'))
File outputFile = new File(pathOutput)
// def outputPath = "M:/measurements.tsv"
// def outputFile = new File(outputPath)

// Create the measurementExporter and start the export
def exporter  = new MeasurementExporter()
                  .imageList(imagesToExport)            // Images from which measurements will be exported
                  .separator(separator)                 // Character that separates values
                  .exportType(exportType)               // Type of objects to export
                  .exportMeasurements(outputFile)        // Start the export process

print "Done!"

print("Done")


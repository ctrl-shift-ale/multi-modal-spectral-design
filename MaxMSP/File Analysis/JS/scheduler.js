autowatch = 1

var FRAMESIZE = 4096 ;

var BUF_SOURCE = new Buffer ("source");
var BUF_OUT = new Buffer ("outBuf"); // 2 audio ch, rms, centroid

var VERBOSE = false;

// buffer channels
var CH_AUDIO_L = 1;
var CH_AUDIO_R = 2;
var CH_RMS = 3;
var CH_CENTROID = 4;
var CH_FFT_AMP = 5;
var CH_FFT_PHASE = 6;

// file in / out folders
var sourcePath = "";
var outputPath = "";
var currentFileName = "";

var filesAnal = [];
var frames = 0;

// parameters received from pfft
var SR = 44100;
var FFTSIZE = 4096;
var VECTORSIZE = 2048;
var FFTHOP = 4096;

// SETTING FUNCTIONS
function source_path(s) {
	sourcePath = s;
}

function out_path(s) {
	outputPath = s;
}

function fft_pars() {
	SR = arguments[0];
	FFTSIZE = arguments[1];
	VECTORSIZE = arguments[2];
	FFTHOP = arguments[3];

}

function main() {

	var sourceFolder = new Folder(sourcePath);
	sourceFolder.typelist = ["WAVE","AIFF"];
	var filesIn = [];
	while (!sourceFolder.end) {
		filesIn.push(sourceFolder.filename);
		sourceFolder.next();
	}
    if (VERBOSE) {
        post("FILES IN: ",filesIn, "\n");
    }
	var outFolder = new Folder(outputPath);
	outFolder.typelist =  ["WAVE","AIFF"];
	var filesOut = [];
	while (!outFolder.end) {
		filesOut.push(outFolder.filename);
		outFolder.next();
	}

	filesAnal.length = 0;
	for (var i = 0; i < filesIn.length; i++) {
		if (!(filesOut.indexOf(filesIn[i]) >= 0)) {
			filesAnal.push(filesIn[i]);
		}
	}

	if (filesAnal.length == 0) {
		post("NO FILES TO ANALYSE FOUND\n");
	} else {
        if (VERBOSE) {
		    post("FILES FOUND: ", filesAnal, "\n");
        }
		analyse();
	}
}

function analyse() {
	currentFileName = filesAnal.shift();
    if (VERBOSE) {
	    post("READ file: ", sourcePath + currentFileName,"\n");
    }
	
	BUF_SOURCE.send("read", sourcePath  + currentFileName);
	var nCh = BUF_SOURCE.channelcount();
	frames = BUF_SOURCE.framecount() / FRAMESIZE ;
	BUF_OUT.send("sizeinsamps", BUF_SOURCE.framecount()*4);
	BUF_OUT.poke(CH_AUDIO_L,0,BUF_SOURCE.peek(CH_AUDIO_L,0,BUF_SOURCE.framecount()));
	var chToCopy = (nCh == 1) ? CH_AUDIO_L : CH_AUDIO_R;	
	BUF_OUT.poke(CH_AUDIO_R,0,BUF_SOURCE.peek(chToCopy,0,BUF_SOURCE.framecount()));
	start_play();
	
}

// this function makes a [fromsymbol] Max object put out a message to a [groove] object 
function start_play() {
    this.patcher.getnamed("js_to_groove").message("0");
}

function end_play() {
	arrRMS = [];
	arrCentroid = [];
	var i = 0;

	while (i < BUF_SOURCE.framecount() - (BUF_SOURCE.framecount() % FRAMESIZE)) {
		
		sumRms = 0; 
		sumCentroid = 0;
		sumCentroid_cnt = 0;
		//post(i, " to ", i + FRAMESIZE, "\n")
		for (var j = i; j < i + FRAMESIZE; j++) {
			sumRms += BUF_SOURCE.peek(1,j) * BUF_SOURCE.peek(1,j);
			/*
			sumCentroid += centrAnal.peek(1,j);
			if (centrAnal.peek[1,j] != 0) {
				sumCentroid_cnt ++
			}
			*/
		}
		arrRMS.push(Math.sqrt(sumRms / FRAMESIZE));
		//arrCentroid.push(sumCentroid / sumCentroid_cnt);

		i += FRAMESIZE;
	} 
    if (VERBOSE) {
	    post("RMS: ", JSON.stringify(arrRMS),"\n");
    }
	BUF_OUT.poke(CH_RMS,0,arrRMS);
	//BUF_OUT.poke(CH_CENTROID,0,arrCentroid);

	BUF_OUT.send("write", outputPath + currentFileName);

	if (filesAnal.length == 0) {
		post("ANALYSIS COMPLETED\n");
	} else {
		analyse();
	}

}

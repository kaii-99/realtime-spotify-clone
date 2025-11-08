import mongoose from "mongoose";

const songSchema = new mongoose.Schema(
	{
		title: {
			type: String,
			required: true,
		},
		artist: {
			type: String,
			required: true,
		},
		imageUrl: {
			type: String,
			required: true,
		},
		audioUrl: {
			type: String,
			required: true,
		},
		duration: {
			type: Number,
			required: true,
		},
		albumId: {
			type: mongoose.Schema.Types.ObjectId,
			ref: "Album",
			required: false,
		},
		genre: {
    	  	type: String,
    	  	required: true, // or false if optional
    	  	enum: ["Pop", "Rock", "Hip-Hop", "Indie","Ballad","Jazz", "Classical", "Electronic", "Other"], // example
    	},
    	language: {
    	  	type: String,
    	  	required: true, // or false if optional
    	  	enum: ["English", "Mandarin", "Malay", "Tamil", "Japanese", "Korean", "Other"], // example
    	},
    	type: {
    	  	type: String,
    	  	required: true,
    	  	enum: ["Normal", "Instrumental"],
    	},
	},
	{ timestamps: true }
);

export const Song = mongoose.model("Song", songSchema);

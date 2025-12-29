import mongoose from "mongoose";

const listeningHistorySchema = new mongoose.Schema(
	{
		userId: {
			type: mongoose.Schema.Types.ObjectId,
			ref: "User",
			required: true,
		},
		songId: {
			type: mongoose.Schema.Types.ObjectId,
			ref: "Song",
			required: true,
		},
		timestamp: {
			type: Date,
			default: Date.now,
		},
		weather: { type: String }, 
  		timeOfDay: { type: String }
	},
	{ timestamps: true }
);

export const ListeningHistory = mongoose.model("ListeningHistory", listeningHistorySchema, "listeninghistories");

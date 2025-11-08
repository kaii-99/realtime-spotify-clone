import { ListeningHistory } from "../models/listeningHistory.model.js";
import { Song } from "../models/song.model.js";

export const exportListeningData = async (req, res, next) => {
	try {
		// Fetch all listening histories
		const histories = await ListeningHistory.find().lean();

		// Fetch all songs (for mapping)
		const songs = await Song.find().lean();
		const songMap = new Map(songs.map(song => [song._id.toString(), song]));

		// Merge both datasets
		const dataset = histories.map(h => {
			const song = songMap.get(h.songId.toString());
			return {
				user_id: h.userId,
				song_id: h.songId,
				timestamp: h.timestamp,
				artist: song?.artist || null,
				title: song?.title || null,
				genre: song?.genre || null,
				language: song?.language || null,
				type: song?.type || null,
			};
		});

		res.status(200).json(dataset);
	} catch (error) {
		console.error("Error exporting listening data:", error);
		res.status(500).json({ message: "Failed to export listening data" });
	}
};

export const exportAllSongData = async (req, res, next) => {
	try {
		// -1 = Descending => newest -> oldest
		// 1 = Ascending => oldest -> newest
		const songs = await Song.find().sort({ createdAt: -1 });
		res.json(songs);
	} catch (error) {
		next(error);
	}
};
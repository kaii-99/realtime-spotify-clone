import { ListeningHistory } from "../models/listeningHistory.model.js";
import { Song } from "../models/song.model.js";
import GroupPlaylist from "../models/groupPlaylist.model.js";

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

export const exportMoodData = async (req, res, next) => {
	try {
	    const histories = await ListeningHistory.find().lean();
	    const songs = await Song.find().lean();

	    const songMap = new Map(
	      	songs.map(song => [song._id.toString(), song])
	    );

	    const dataset = histories.map(h => {
	      	const song = songMap.get(h.songId.toString());

	      	return {
	      	  	user_id: h.userId.toString(),
	      	  	song_id: h.songId.toString(),
				
	      	  	genre: song?.genre || null,
	      	  	language: song?.language || null,
	      	  	type: song?.type || null,
				
	      	  	weather: h.weather || null,
	      	  	timeOfDay: h.timeOfDay || null,
				
	      	  	timestamp: h.timestamp
	      	};
	    });

	    res.status(200).json(dataset);

	} catch (error) {
	  	console.error("Error exporting listening data:", error);
	  	res.status(500).json({ message: "Failed to export listening data" });
	}
}

export const exportMoodData_Personalized = async (req, res, next) => {
	try {
		const userId = req.params.userId;
		
	    const histories = await ListeningHistory.find({ userId }).lean();
	    const songs = await Song.find().lean();

	    const songMap = new Map(
	      	songs.map(song => [song._id.toString(), song])
	    );

	    const dataset = histories.map(h => {
	      	const song = songMap.get(h.songId.toString());

	      	return {
	      	  	user_id: h.userId.toString(),
	      	  	song_id: h.songId.toString(),
				
	      	  	genre: song?.genre || null,
	      	  	language: song?.language || null,
	      	  	type: song?.type || null,
				
	      	  	weather: h.weather || null,
	      	  	timeOfDay: h.timeOfDay || null,
				
	      	  	timestamp: h.timestamp
	      	};
	    });

	    res.status(200).json(dataset);

	} catch (error) {
	  	console.error("Error exporting listening data:", error);
	  	res.status(500).json({ message: "Failed to export listening data" });
	}
}

export const fetchUserGroupPlaylists = async (req, res) => {
  try {
	const { groupId } = req.params;

	const group = await GroupPlaylist.findById(groupId)
      .select("members")   // ONLY fetch members
      .lean();

    if (!group) {
      return res.status(404).json({ message: "Group playlist not found" });
    }

    res.status(200).json({
      groupId,
      members: group.members   // array of userIds
    });

  } catch (error) {
    console.error("Fetch group members error:", error);
    res.status(500).json({ message: "Failed to fetch group members" });
  }
};
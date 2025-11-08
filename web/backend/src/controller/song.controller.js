import { Song } from "../models/song.model.js";

export const getAllSongs = async (req, res, next) => {
	try {
		// -1 = Descending => newest -> oldest
		// 1 = Ascending => oldest -> newest
		const songs = await Song.find().sort({ createdAt: -1 });
		res.json(songs);
	} catch (error) {
		next(error);
	}
};

export const getAllSongsPublic = async (req, res, next) => {
	try {
		// -1 = Descending => newest -> oldest
		// 1 = Ascending => oldest -> newest
		const songs = await Song.find().sort({ createdAt: -1 });
		res.json(songs);
	} catch (error) {
		next(error);
	}
};

export const getFeaturedSongs = async (req, res, next) => {
	try {
		// fetch 6 random songs using mongodb's aggregation pipeline
		const songs = await Song.aggregate([
			{
				$sample: { size: 6 },
			},
			{
				$project: {
					_id: 1,
					title: 1,
					artist: 1,
					imageUrl: 1,
					audioUrl: 1,
				},
			},
		]);

		res.json(songs);
	} catch (error) {
		next(error);
	}
};

export const getMadeForYouSongs = async (req, res, next) => {
	try {
		const songs = await Song.aggregate([
			{
				$sample: { size: 4 },
			},
			{
				$project: {
					_id: 1,
					title: 1,
					artist: 1,
					imageUrl: 1,
					audioUrl: 1,
				},
			},
		]);

		res.json(songs);
	} catch (error) {
		next(error);
	}
};

export const getTrendingSongs = async (req, res, next) => {
	try {
		const songs = await Song.aggregate([
			{
				$sample: { size: 4 },
			},
			{
				$project: {
					_id: 1,
					title: 1,
					artist: 1,
					imageUrl: 1,
					audioUrl: 1,
				},
			},
		]);

		res.json(songs);
	} catch (error) {
		next(error);
	}
};

export const getRecommendedSongs = async (req, res, next) => {
  try {
    const userId = req.userId; // assuming Clerk or JWT sets this

    // 1️⃣ Get the user's recent listening history
    const recentHistory = await ListeningHistory.find({ userId })
      .sort({ createdAt: -1 })
      .limit(5)
      .populate("songId");

    if (recentHistory.length === 0) {
      // fallback to random if no history
      const randomSongs = await Song.aggregate([{ $sample: { size: 6 } }]);
      return res.json(randomSongs);
    }

    // 2️⃣ Extract genres/languages/types from recent songs
    const recentGenres = recentHistory.map((h) => h.songId.genre);
    const recentLanguages = recentHistory.map((h) => h.songId.language);

    // 3️⃣ Find songs with similar attributes
    const recommended = await Song.find({
      $or: [
        { genre: { $in: recentGenres } },
        { language: { $in: recentLanguages } },
      ],
    })
      .limit(10);

    res.json(recommended);
  } catch (error) {
    console.error("Error in getRecommendedSongs", error);
    next(error);
  }
};
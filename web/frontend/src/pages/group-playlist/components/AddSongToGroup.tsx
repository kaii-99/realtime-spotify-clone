import { useState, useEffect } from "react";
import { useChatStore } from "@/stores/useChatStore";
import { useMusicStore } from "@/stores/useMusicStore";

const AddSongToGroup = ({
  playlistId,
  refresh,
}: {
  playlistId: string;
  refresh: () => void;
}) => {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);

  const { songs, fetchSongs } = useMusicStore();
  const { AddSongGroupPlaylist } = useChatStore();

  useEffect(() => {
    fetchSongs();
  }, [fetchSongs]);

  const filteredSongs = songs.filter((song: any) =>
    song.title.toLowerCase().includes(query.toLowerCase())
  );

  const handleAdd = async (songId: string) => {
    try {
      setLoading(true);

      await AddSongGroupPlaylist({ groupId: playlistId, songId: songId});

      setQuery("");
      setShowResults(false);
      refresh();

    } catch (err) {
      console.error("Failed to add song:", err);
      alert("Failed to add song");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative w-full">
      <input
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setShowResults(true);
        }}
        onBlur={() => setTimeout(() => setShowResults(false), 150)}
        placeholder="Search song name..."
        className="w-full p-2 rounded bg-zinc-800 text-white border border-zinc-700"
      />

      {showResults && query && (
        <div className="absolute z-20 w-full mt-1 bg-zinc-900 border border-zinc-700 rounded-lg max-h-60 overflow-y-auto">
          {filteredSongs.length === 0 ? (
            <p className="p-2 text-zinc-500">No matching songs</p>
          ) : (
            filteredSongs.map((song: any) => (
              <button
                key={song._id}
                onClick={() => handleAdd(song._id)}
                className="w-full text-left p-2 hover:bg-zinc-800 flex gap-3"
                disabled={loading}
              >
                <div className="flex-1">
                  <p className="text-white">{song.title}</p>
                  <p className="text-sm text-zinc-400">{song.artist}</p>
                </div>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
};

export default AddSongToGroup;

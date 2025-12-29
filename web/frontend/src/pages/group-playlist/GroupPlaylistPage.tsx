import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { axiosInstance } from "@/lib/axios";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useChatStore } from "@/stores/useChatStore";
import { useMusicStore } from "@/stores/useMusicStore";
import PlayButton from "./components/PlayButton";
import AddSongToGroup from "./components/AddSongToGroup";

const GroupPlaylistPage = () => {
  const { id } = useParams();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const { FetchSongGroupPlaylist, groupPlaylistSongs } = useChatStore();
  const { 
    recommendedSongs_groupplaylist, fetchRecommendationGroupPlaylist, 
    recommendedSongs_groupplaylist_DL, fetchRecommendationGroupPlaylist_DL, 
  } = useMusicStore();
  const [recType, setRecType] = useState<"general" | "general_enhanced">(
    "general"
  );

  useEffect(() => {
    if (!id) return;

    const fetchPlaylist = async () => {
      try {
        setLoading(true);

        const data = await FetchSongGroupPlaylist({ groupId: id});        
        console.log("GROUP PLAYLIST DATA: ", data);
      } catch (err) {
        console.error(err);
        setError("Failed to load playlist");
      } finally {
        setLoading(false);
      }
    };

    fetchPlaylist();
  }, [id, FetchSongGroupPlaylist]);

  useEffect(() => {
    if (id) {
      fetchRecommendationGroupPlaylist(id);
      fetchRecommendationGroupPlaylist_DL(id);
    }
  }, [id, fetchRecommendationGroupPlaylist, fetchRecommendationGroupPlaylist_DL]);

  if (loading) return <p className="text-white p-4">Loading...</p>;
  if (error) return <p className="text-red-500 p-4">{error}</p>;
  if (!groupPlaylistSongs) return null;

  const displayedRecommendations =
  recType === "general"
    ? recommendedSongs_groupplaylist
    : recommendedSongs_groupplaylist_DL;

  return (
    <div className="p-6 text-white">
      <h1 className="text-2xl font-bold mb-2">{groupPlaylistSongs.name}</h1>
      <p className="text-zinc-400 mb-6">
        {groupPlaylistSongs.songs.length} songs
      </p>

      <ScrollArea className='h-[calc(100vh-180px)]'>
        <div className="pb-24">
          {/* Recommendation Selector */}
          <div className="mb-6">
            <h2 className="text-lg font-semibold mb-3">
              Select Recommendation Playlist
            </h2>

            <div className="flex gap-3 mb-4">
              <button
                onClick={() => setRecType("general")}
                className={`px-4 py-2 rounded-md border transition
                  ${
                    recType === "general"
                      ? "bg-green-600 border-green-500 text-white"
                      : "bg-zinc-900 border-zinc-700 text-zinc-400 hover:text-white"
                  }`}
              >
                General
              </button>
                
              <button
                onClick={() => setRecType("general_enhanced")}
                className={`px-4 py-2 rounded-md border transition
                  ${
                    recType === "general_enhanced"
                      ? "bg-green-600 border-green-500 text-white"
                      : "bg-zinc-900 border-zinc-700 text-zinc-400 hover:text-white"
                  }`}
              >
                General Enhanced
              </button>
            </div>
                
            {displayedRecommendations.length === 0 ? (
              <p className="text-zinc-500">No songs yet. Add one 👇</p>
            ) : (
              <div className="space-y-3">
                {displayedRecommendations.map((song) => (
                  <div
                    key={song._id}
                    className="relative group p-3 bg-zinc-900 rounded-md flex justify-between items-center"
                  >
                    <div>
                      <p className="font-medium">{song.title}</p>
                      <p className="text-sm text-zinc-400">{song.artist}</p>
                    </div>
                    <PlayButton song={song} />
                  </div>
                ))}
              </div>
            )}

          </div>

          <h2 className="text-lg font-semibold mb-4">Songs</h2>

          {groupPlaylistSongs.songs.length === 0 ? (
            <p className="text-zinc-500">No songs yet. Add one 👇</p>
          ) : (
            <div className="space-y-3">
              {groupPlaylistSongs.songs.map((song: any) => (
                <div
                  key={song._id}
                  className="p-3 bg-zinc-900 rounded-md flex justify-between items-center"
                >
                  <div>
                    <p className="font-medium">{song.title}</p>
                    <p className="text-sm text-zinc-400">{song.artist}</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Add Song Area */}
          <div className="mt-8">
            <h3 className="font-semibold mb-2">Add a new song</h3>
            <AddSongToGroup
              playlistId={id!}
              refresh={() => FetchSongGroupPlaylist({ groupId: id! })}
            />
          </div>
        </div>
      </ScrollArea>
    </div>
  );
};

export default GroupPlaylistPage;

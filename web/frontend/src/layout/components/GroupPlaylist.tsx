import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useChatStore } from "@/stores/useChatStore";
import { useUser } from "@clerk/clerk-react";

const GroupPlaylists = () => {
    const [loading, setLoading] = useState(false);
    const { user } = useUser();
    const { fetchGroupPlaylist, groupPlaylists } = useChatStore();

    useEffect(() => {
        if (!user) return;

        const loadData = async () => {
            try {
                setLoading(true);
                console.log("Group Playlist fetch user:",user.id)
                const data = await fetchGroupPlaylist({ user: user.id});
                
                console.log("GROUP PLAYLIST DATA: ", data);
            } catch (err) {
                console.error("Error loading group playlists:", err);
            } finally {
                setLoading(false);
            }
        };

        loadData();

    }, [user, fetchGroupPlaylist]);

    if (loading) {
        return <p className="text-white px-2">Loading group playlists...</p>;
    }

    if (groupPlaylists.length === 0) {
        return <p className="text-zinc-400 px-2">No group playlists yet</p>;
    }

    return (
        <div className="space-y-2">
            {groupPlaylists.map((playlist) => (
                <Link
                    to={`/group-playlist/${playlist._id}`}
                    key={playlist._id}
                    className="p-2 hover:bg-zinc-800 rounded-md flex flex-col cursor-pointer"
                >
                    <p className="font-medium text-white truncate">
                        {playlist.name}
                    </p>
                </Link>
            ))}
        </div>
    );
};

export default GroupPlaylists;

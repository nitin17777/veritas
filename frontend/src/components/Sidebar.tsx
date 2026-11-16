import type { Page, SettingsConfig } from "../types";
import { Icon } from "./Icons";
import { Logo } from "./Logo";

interface SidebarProps {
  page: Page;
  setPage: (p: Page) => void;
  settings?: SettingsConfig;
}

export function Sidebar({ page, setPage, settings }: { page: Page; setPage: (p: Page) => void; settings?: SettingsConfig }) {
  const links: { id: Page; label: string; icon: string }[] = [
    { id: "overview", label: "Overview", icon: "overview" },
    { id: "verify", label: "Verify a claim", icon: "verify" },
    { id: "history", label: "Verification history", icon: "history" },
    { id: "research", label: "Research / About", icon: "research" },
    { id: "settings", label: "Settings", icon: "settings" },
  ];

  const active = page === "processing" || page === "results" ? "verify" : page;
  const userName = settings?.userName || "Alex Morgan";
  const workspaceName = settings?.workspaceName || "Research workspace";
  const initials = userName
    .split(" ")
    .map((n) => n[0])
    .join("")
    .slice(0, 2)
    .toUpperCase() || "AM";

  return (
    <aside className="sidebar">
      <div className="sidebar-top">
        <Logo />
        <nav aria-label="Primary navigation">
          <span className="nav-label">Workspace</span>
          {links.map((link) => (
            <button
              key={link.id}
              className={`nav-item ${active === link.id ? "active" : ""}`}
              onClick={() => setPage(link.id)}
            >
              <Icon name={link.icon} />
              <span>{link.label}</span>
            </button>
          ))}
        </nav>
      </div>
      <div className="sidebar-bottom">
        <button className="nav-item" onClick={() => setPage("research")}>
          <Icon name="help" />
          <span>Help & documentation</span>
        </button>
        <div className="profile" onClick={() => setPage("settings")} style={{ cursor: "pointer" }}>
          <div className="avatar">{initials}</div>
          <div>
            <strong>{userName}</strong>
            <span>{workspaceName}</span>
          </div>
          <Icon name="more" size={18} />
        </div>
      </div>
    </aside>
  );
}

import { Icon } from "./Icons";

interface LogoProps {
  onClick?: () => void;
}

export function Logo({ onClick }: LogoProps = {}) {
  return (
    <div
      className="logo"
      onClick={onClick}
      style={onClick ? { cursor: "pointer" } : undefined}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={onClick ? (e) => e.key === "Enter" && onClick() : undefined}
      aria-label={onClick ? "Go to home page" : undefined}
    >
      <span className="logo-mark">
        <Icon name="check" size={15} />
      </span>
      <span>VERITAS</span>
    </div>
  );
}

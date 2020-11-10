----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 10.12.2019 15:01:26
-- Design Name: 
-- Module Name: read_byte - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments: possibile usare funzioni per leggere e scrivere byte
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity read_byte is
    Port ( clk_in : in STD_LOGIC;
           clk_ph : in STD_LOGIC;
           rstn : in STD_LOGIC;
           strt: in STD_LOGIC;  -- read enable
           pin_in : in STD_LOGIC;
           error, busy, readable : out STD_LOGIC;   --forse basta solo uno tra busy e readable
           word_out : out STD_LOGIC_VECTOR (7 downto 0) 
           );
end read_byte;

architecture Behavioral of read_byte is


--signal clk1, clk2 : STD_LOGIC;
signal error_signal, busy_signal : STD_LOGIC;
signal word_buff : STD_LOGIC_VECTOR ( 7 downto 0 ) := x"00";
type state is (s_wait_low, s_read, s_high, s_error);
signal state_fsm: state;

begin


    --process (clk_ph, clk_in, clk1, clk2) --not necessary
    --begin
        --if clk_ph'event then
            --clk2 <= clk_in ;
            --clk1 <= clk2 ;
        --end if ;
    
    --end process ;
    
    read : process (clk_ph, strt, pin_in) is
    
    variable word_cnt : natural:= 0;
    variable word : STD_LOGIC_VECTOR (7 downto 0 ) ;
    
    begin
        if rstn = '0' then
            state_fsm <= s_wait_low ;
        elsif rising_edge(clk_ph) then   --an asynchronous reset would be nice
            case state_fsm is
            when s_wait_low =>
                if strt = '1' then
                    if pin_in = '0' then
                        state_fsm <= s_read ;
                        --busy_signal <= '1' ;
                    else
                        state_fsm <= s_wait_low ;
                        --busy_signal <= '0'; 
                        --readable <= '0' ;
                    end if;
                else
                    state_fsm <= s_wait_low ;
                    --busy_signal <= '0'; 
                    --readable <= '0' ;    
                end if;
            when s_read =>
                --readable <= '0';
                --busy_signal <= '1';
                word(word_cnt) := pin_in;
                if word_cnt = 7 then	--whole word read
                    word_buff <= word;
				    state_fsm <= s_high;	--change state
				    word_cnt := 0; --reset counter word
				else
				    word_cnt := word_cnt + 1;
				end if;
		     when s_high =>
		         if pin_in = '1' then
                     --readable <= '1';
                     state_fsm <= s_wait_low;
                     --busy_signal <= '0';
				 else
				     state_fsm <= s_error;
				     --error_signal <= '1';   
				 end if;
			  when s_error =>
			     null ;
              when others =>
                  state_fsm <= s_wait_low;
                end case;
            else 
                null;    
        end if;
        
    end process;
    
    process (clk_in) is
    begin
        
        if rising_edge(clk_in) then
            case state_fsm is
            when s_wait_low =>
                readable <= '0' ;
                busy_signal <= '0' ;
                error_signal <= '0' ;
            when s_read =>
                readable <= '0' ;
                busy_signal <= '1' ;
                error_signal <= '0' ;
            when s_high =>
                readable <= '1' ;
                busy_signal <= '0' ;
                error_signal <= '0' ;
            when s_error =>
                readable <= '0' ;
                busy_signal <= '0' ;
                error_signal <= '1' ;
            when others =>
                null ;
            end case;
        end if;
    end process;
    
    word_out <= word_buff ;
    busy <= busy_signal ;
    error <= error_signal ;
    
end Behavioral;

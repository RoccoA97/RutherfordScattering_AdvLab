`timescale 1ns / 1ps


module ROM_INIT(
    input [7:0] addra,
    input clka,
    input ena,
    output [7:0] douta
    );
    
    // xpm_memory_sprom: Single Port ROM
    // Xilinx Parameterized Macro, version 2018.3
    xpm_memory_sprom #(
    .ADDR_WIDTH_A(8), // DECIMAL
    .AUTO_SLEEP_TIME(0), // DECIMAL
    .ECC_MODE("no_ecc"), // String
    .MEMORY_INIT_FILE("ROM_initialization.mem"), // String
    .MEMORY_INIT_PARAM(""), // String
    .MEMORY_OPTIMIZATION("true"), // String
    .MEMORY_PRIMITIVE("auto"), // String
    .MEMORY_SIZE(2048), // DECIMAL
    .MESSAGE_CONTROL(0), // DECIMAL
    .READ_DATA_WIDTH_A(8), // DECIMAL
    .READ_LATENCY_A(2), // DECIMAL
    .READ_RESET_VALUE_A("0"), // String
    .RST_MODE_A("SYNC"), // String
    .USE_MEM_INIT(1), // DECIMAL
    .WAKEUP_TIME("disable_sleep") // String
    )
    xpm_memory_sprom_inst (
    .dbiterra(), // 1-bit output: Leave open.
    .douta(douta), // READ_DATA_WIDTH_A-bit output: Data output for port A read operations.
    .sbiterra(), // 1-bit output: Leave open.
    .addra(addra), // ADDR_WIDTH_A-bit input: Address for port A read operations.
    .clka(clka), // 1-bit input: Clock signal for port A.
    .ena(ena), // 1-bit input: Memory enable signal for port A. Must be high on clock
    // cycles when read operations are initiated. Pipelined internally.
    //.injectdbiterra(injectdbiterra), // 1-bit input: Do not change from the provided value.
    //.injectsbiterra(injectsbiterra), // 1-bit input: Do not change from the provided value.
    .regcea(1'b1), // 1-bit input: Do not change from the provided value.
    .rsta(rsta) // 1-bit input: Reset signal for the final port A output register stage.
    // Synchronously resets output port douta to the value specified by
    // parameter READ_RESET_VALUE_A.
    //.sleep(sleep) // 1-bit input: sleep signal to enable the dynamic power saving feature.
    );
    // End of xpm_memory_sprom_inst instantiation
    
endmodule
